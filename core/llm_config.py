"""
LLM-driven Hyperparameter Configurator
Uses local llama.cpp (no external server).
Hardware scan includes: VRAM, RAM, CPU, GPU name, CUDA, disk.
Falls back to rule-based logic if llama.cpp is unavailable or returns bad JSON.
"""

import json
import os
import re
import shutil
import subprocess
import urllib.request

from llama_cpp import Llama

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gemma-2b-q4.gguf")
MODEL_URL = "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_0.gguf"
MIN_MODEL_BYTES = 50 * 1024 * 1024

SAFE_LIMITS = {
    "batch_size":          {"min": 1,       "max": 32},
    "epochs":              {"min": 100,     "max": 300},
    "learning_rate":       {"min": 0.0001,  "max": 0.0001},
    "save_every_n_epochs": {"min": 10,      "max": 10},
}

SYSTEM_PROMPT = """You are a hyperparameter generation engine.

You MUST return ONLY valid JSON.
DO NOT explain anything.
DO NOT add extra text.
DO NOT use markdown.

INPUT:
VRAM_GB: {vram}
RAM_GB: {ram}
DATASET_MINUTES: {minutes}
SEGMENT_COUNT: {segments}

RULES:

1. batch_size:
   = min(32, max(1, int(VRAM_GB * 2)))

2. epochs:
   if DATASET_MINUTES < 5 → 250
   elif DATASET_MINUTES < 10 → 200
   elif DATASET_MINUTES < 20 → 150
   else → 120

3. learning_rate:
   = 0.0001

4. save_every_n_epochs:
   = 10

OUTPUT FORMAT:

{{
  "batch_size": <integer>,
  "epochs": <integer>,
  "learning_rate": <float>,
  "save_every_n_epochs": <integer>
}}"""


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output")
    return json.loads(match.group(0))


def _ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) >= MIN_MODEL_BYTES:
            return
        print("[LLM] Model file too small. Re-downloading...")
        os.remove(MODEL_PATH)
    print("[LLM] Model not found. Downloading...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[LLM] Model downloaded.")


class LLMConfigurator:

    def get_hyperparameters(self, dataset_stats: dict) -> dict:
        hw = self._scan_hardware()
        print(f"[LLMConfig] Hardware: VRAM={hw['vram_mb']}MB "
              f"RAM={hw['ram_mb']}MB CPU={hw['cpu_cores']} "
              f"GPU={hw['gpu_name']} CUDA={hw['cuda_available']}")

        try:
            params = self._query_llama(hw, dataset_stats)
            params = self._validate_and_clamp(params)
            print(f"[LLMConfig] Generated hyperparameters: {params}")
            return params
        except Exception as e:
            print(f"[LLMConfig] Llama.cpp failed ({e!r}), using rule-based fallback.")
            params = self._rule_based_fallback(hw, dataset_stats)
            print(f"[LLMConfig] Fallback hyperparams: {params}")
            return params

    # ------------------------------------------------------------------
    # Hardware scanning
    # ------------------------------------------------------------------

    def _scan_hardware(self) -> dict:
        vram_mb, gpu_name = self._get_gpu_info()
        cuda_available    = self._check_cuda()
        return {
            "vram_mb":        vram_mb,
            "ram_mb":         self._get_ram_mb(),
            "cpu_cores":      os.cpu_count() or 4,
            "gpu_name":       gpu_name,
            "cuda_available": cuda_available,
            "disk_free_gb":   self._get_disk_free_gb(),
        }

    def _get_gpu_info(self) -> tuple:
        try:
            vram_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip().splitlines()
            name_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name",
                 "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip().splitlines()
            if vram_out:
                return int(vram_out[0].strip()), name_out[0].strip() if name_out else "NVIDIA GPU"
        except Exception:
            pass
        return 0, "CPU-only"

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _get_ram_mb(self) -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            return 8192

    def _get_disk_free_gb(self) -> float:
        try:
            usage = shutil.disk_usage(".")
            return round(usage.free / (1024 ** 3), 1)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # llama.cpp query
    # ------------------------------------------------------------------

    def _query_llama(self, hw: dict, dataset_stats: dict) -> dict:
        vram_gb = round(hw.get("vram_mb", 0) / 1024.0, 2)
        ram_gb = round(hw.get("ram_mb", 0) / 1024.0, 2)
        minutes = round(dataset_stats.get("total_duration_minutes", 0), 2)
        segments = int(dataset_stats.get("segment_count", 0))

        prompt = SYSTEM_PROMPT.format(
            vram=vram_gb,
            ram=ram_gb,
            minutes=minutes,
            segments=segments,
        )
        wrapped_prompt = (
            "<start_of_turn>user\n"
            f"{prompt}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        _ensure_model()
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=512,
            n_threads=4,
            verbose=False,
        )
        print("[LLM] Model loaded (llama.cpp)")
        response = llm(
            wrapped_prompt + "{",
            max_tokens=120,
            temperature=0.1,
            stop=["}", "<end_of_turn>"],
        )
        if not isinstance(response, dict) or "choices" not in response:
            raise ValueError(f"Unexpected llama.cpp response: {response!r}")
        text = "{" + response["choices"][0].get("text", "") + "}"
        try:
            return extract_json(text)
        except Exception:
            return self._rule_based_fallback(hw, dataset_stats)

    # ------------------------------------------------------------------
    # Validation & clamping
    # ------------------------------------------------------------------

    def _validate_and_clamp(self, params: dict) -> dict:
        clamped = {}
        for key, limits in SAFE_LIMITS.items():
            val = float(params[key])
            val = max(limits["min"], min(limits["max"], val))
            clamped[key] = val
        clamped["batch_size"]          = int(clamped["batch_size"])
        clamped["epochs"]              = int(clamped["epochs"])
        clamped["learning_rate"]       = float(clamped["learning_rate"])
        clamped["save_every_n_epochs"] = int(clamped["save_every_n_epochs"])
        return clamped

    # ------------------------------------------------------------------
    # Rule-based fallback — FIXED
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, hw: dict, dataset_stats: dict) -> dict:
        vram_gb       = hw.get("vram_mb", 0) / 1024.0
        total_minutes = dataset_stats.get("total_duration_minutes", 0)

        if total_minutes < 5:
            epochs = 250
        elif total_minutes < 10:
            epochs = 200
        elif total_minutes < 20:
            epochs = 150
        else:
            epochs = 120

        batch_size = min(32, max(1, int(vram_gb * 2)))
        return {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": 0.0001,
            "save_every_n_epochs": 10,
        }
