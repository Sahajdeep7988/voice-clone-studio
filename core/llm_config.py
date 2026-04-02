"""
LLM-driven Hyperparameter Configurator
Uses Gemma 3 1B via Ollama (localhost:11434).
Hardware scan includes: VRAM, RAM, CPU, GPU name, CUDA, disk.
Falls back to rule-based logic if Ollama is unavailable or returns bad JSON.
"""

import json
import os
import re
import shutil
import subprocess
import requests


OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

SAFE_LIMITS = {
    "batch_size":          {"min": 1,       "max": 32},
    "epochs":              {"min": 100,     "max": 1000},
    "learning_rate":       {"min": 0.00001, "max": 0.001},
    "save_every_n_epochs": {"min": 1,       "max": 100},
}

SYSTEM_PROMPT = """You are an expert in RVC (Retrieval-based Voice Conversion) model training.
Given detailed hardware specs and dataset statistics, choose optimal training hyperparameters.

Rules:
- batch_size: scale with VRAM (1 per ~512 MB VRAM, max 32). Reduce if dataset is small (<200 segments).
- epochs: more data → fewer epochs needed. Poor quality (<10 min) → 500+. Optimal (>20 min) → 300.
- learning_rate: default 0.0001. Increase to 0.0002 for CPU-only or very small datasets.
- save_every_n_epochs: ~10% of total epochs, minimum 10, maximum 100.
- If CUDA is unavailable, set batch_size=2 and epochs≤200.

Return ONLY valid JSON with exactly these keys (no markdown, no explanation):
{
  "batch_size": <integer 1-32>,
  "epochs": <integer 100-1000>,
  "learning_rate": <float 0.00001-0.001>,
  "save_every_n_epochs": <integer 1-100>
}"""


class LLMConfigurator:

    def get_hyperparameters(self, dataset_stats: dict) -> dict:
        hw = self._scan_hardware()
        print(f"[LLMConfig] Hardware: VRAM={hw['vram_mb']}MB "
              f"RAM={hw['ram_mb']}MB CPU={hw['cpu_cores']} "
              f"GPU={hw['gpu_name']} CUDA={hw['cuda_available']}")

        try:
            params = self._query_ollama(hw, dataset_stats)
            params = self._validate_and_clamp(params)
            print(f"[LLMConfig] Ollama hyperparams: {params}")
            return params
        except Exception as e:
            print(f"[LLMConfig] Ollama failed ({e}), using rule-based fallback.")
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
        """Returns (vram_mb: int, gpu_name: str)"""
        # NVIDIA
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

        # AMD ROCm
        try:
            out = subprocess.check_output(
                ["rocminfo"], stderr=subprocess.DEVNULL, text=True
            )
            m_mem  = re.search(r"Memory Size.*?:\s*(\d+)\s*\(", out)
            m_name = re.search(r"Marketing Name:.*?(\w[\w\s]+)", out)
            if m_mem:
                return int(m_mem.group(1)), m_name.group(1).strip() if m_name else "AMD GPU"
        except Exception:
            pass

        return 0, "CPU-only"

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["python3", "-c", "import torch; print(torch.cuda.is_available())"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            return out == "True"
        except Exception:
            return False

    def _get_ram_mb(self) -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
        return 8192

    def _get_disk_free_gb(self) -> float:
        try:
            usage = shutil.disk_usage(".")
            return round(usage.free / (1024 ** 3), 1)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Ollama query
    # ------------------------------------------------------------------

    def _query_ollama(self, hw: dict, dataset_stats: dict) -> dict:
        avg_dur = (
            dataset_stats.get("total_duration_minutes", 0) * 60
            / max(dataset_stats.get("segment_count", 1), 1)
        )
        context = {
            "hardware": hw,
            "dataset": {
                **dataset_stats,
                "avg_segment_duration_s": round(avg_dur, 1),
            },
        }
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{json.dumps(context, indent=2)}"

        response = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.1, "num_predict": 150},
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        return self._extract_json(raw)

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # Strip markdown code blocks
        text = re.sub(r"```[a-z]*\n?", "", text)
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"No valid JSON in Ollama response: {text!r}")

    # ------------------------------------------------------------------
    # Validation & clamping
    # ------------------------------------------------------------------

    def _validate_and_clamp(self, params: dict) -> dict:
        required = set(SAFE_LIMITS.keys())
        missing  = required - params.keys()
        if missing:
            raise ValueError(f"Missing keys in Ollama response: {missing}")
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
    # Rule-based fallback — dataset-aware
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, hw: dict, dataset_stats: dict) -> dict:
        vram_gb       = hw.get("vram_mb", 0) / 1024.0
        cuda          = hw.get("cuda_available", False)
        total_minutes = dataset_stats.get("total_duration_minutes", 0)

        if not cuda:
            return {"batch_size": 2, "epochs": 200,
                    "learning_rate": 0.0002, "save_every_n_epochs": 20}

        # Epoch count driven by dataset size
        if total_minutes >= 20:
            epochs = 300
        elif total_minutes >= 10:
            epochs = 400
        else:
            epochs = 500

        save_every = max(10, min(100, epochs // 10))

        if vram_gb >= 8:
            return {"batch_size": 16, "epochs": epochs,
                    "learning_rate": 0.0001, "save_every_n_epochs": save_every}
        elif vram_gb >= 6:
            return {"batch_size": 8, "epochs": epochs,
                    "learning_rate": 0.0001, "save_every_n_epochs": save_every}
        elif vram_gb >= 4:
            return {"batch_size": 4, "epochs": epochs,
                    "learning_rate": 0.0001, "save_every_n_epochs": save_every}
        else:
            return {"batch_size": 2, "epochs": min(epochs, 300),
                    "learning_rate": 0.0002, "save_every_n_epochs": save_every}
