"""
LLM-driven Hyperparameter Configurator
Uses Gemma 3 1B via Ollama (localhost:11434) to pick training params.
Falls back to rule-based logic if Ollama is unavailable or returns bad JSON.
"""

import json
import os
import re
import subprocess
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

SAFE_LIMITS = {
    "batch_size":       {"min": 1,       "max": 32},
    "epochs":           {"min": 100,     "max": 1000},
    "learning_rate":    {"min": 0.00001, "max": 0.001},
    "save_every_n_epochs": {"min": 1,    "max": 100},
}


class LLMConfigurator:

    def get_hyperparameters(self, dataset_stats: dict) -> dict:
        """
        Returns training hyperparameters as a dict.
        Tries Ollama first, falls back to rule-based logic.
        """
        hw = self._scan_hardware()
        print(f"[LLMConfig] Hardware: {hw}")

        try:
            params = self._query_ollama(hw, dataset_stats)
            params = self._validate_and_clamp(params)
            print(f"[LLMConfig] Ollama hyperparams: {params}")
            return params
        except Exception as e:
            print(f"[LLMConfig] Ollama failed ({e}), using rule-based fallback.")
            params = self._rule_based_fallback(hw)
            print(f"[LLMConfig] Fallback hyperparams: {params}")
            return params

    # ------------------------------------------------------------------
    # Hardware scanning
    # ------------------------------------------------------------------

    def _scan_hardware(self) -> dict:
        vram_mb = self._get_vram_mb()
        ram_mb = self._get_ram_mb()
        cpu_cores = os.cpu_count() or 4
        return {
            "vram_mb": vram_mb,
            "ram_mb": ram_mb,
            "cpu_cores": cpu_cores,
        }

    def _get_vram_mb(self) -> int:
        # Try NVIDIA first
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
            if lines:
                return int(lines[0])
        except Exception:
            pass

        # Try AMD ROCm
        try:
            out = subprocess.check_output(
                ["rocminfo"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            match = re.search(r"Memory Size.*?:\s*(\d+)\s*\(", out)
            if match:
                return int(match.group(1))
        except Exception:
            pass

        return 0  # CPU-only

    def _get_ram_mb(self) -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
        return 8192  # safe default: 8 GB

    # ------------------------------------------------------------------
    # Ollama query
    # ------------------------------------------------------------------

    def _query_ollama(self, hw: dict, dataset_stats: dict) -> dict:
        context = {
            "hardware": hw,
            "dataset": dataset_stats,
        }
        prompt = (
            "You are a machine learning hyperparameter expert.\n"
            "Given the hardware and dataset info below, choose optimal RVC "
            "voice cloning training hyperparameters.\n\n"
            f"Context (JSON):\n{json.dumps(context, indent=2)}\n\n"
            "Return ONLY a valid JSON object with exactly these keys "
            "(no explanation, no markdown, no extra text):\n"
            '{\n'
            '  "batch_size": <integer 1-32>,\n'
            '  "epochs": <integer 100-1000>,\n'
            '  "learning_rate": <float 0.00001-0.001>,\n'
            '  "save_every_n_epochs": <integer 1-100>\n'
            '}'
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")

        # Extract JSON from the response (model may include extra text)
        return self._extract_json(raw)

    def _extract_json(self, text: str) -> dict:
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"No valid JSON found in Ollama response: {text!r}")

    # ------------------------------------------------------------------
    # Validation & clamping
    # ------------------------------------------------------------------

    def _validate_and_clamp(self, params: dict) -> dict:
        required = {"batch_size", "epochs", "learning_rate", "save_every_n_epochs"}
        missing = required - params.keys()
        if missing:
            raise ValueError(f"Missing keys in Ollama response: {missing}")

        clamped = {}
        for key, limits in SAFE_LIMITS.items():
            val = params[key]
            val = max(limits["min"], min(limits["max"], val))
            clamped[key] = val

        # Ensure correct types
        clamped["batch_size"] = int(clamped["batch_size"])
        clamped["epochs"] = int(clamped["epochs"])
        clamped["learning_rate"] = float(clamped["learning_rate"])
        clamped["save_every_n_epochs"] = int(clamped["save_every_n_epochs"])
        return clamped

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_fallback(self, hw: dict) -> dict:
        vram_mb = hw.get("vram_mb", 0)
        vram_gb = vram_mb / 1024.0

        if vram_gb >= 8:
            return {
                "batch_size": 8,
                "epochs": 500,
                "learning_rate": 0.0001,
                "save_every_n_epochs": 50,
            }
        elif vram_gb >= 4:
            return {
                "batch_size": 4,
                "epochs": 300,
                "learning_rate": 0.0001,
                "save_every_n_epochs": 50,
            }
        else:
            return {
                "batch_size": 2,
                "epochs": 200,
                "learning_rate": 0.0002,
                "save_every_n_epochs": 25,
            }
