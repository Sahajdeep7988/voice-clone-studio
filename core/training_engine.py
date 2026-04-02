"""
RVC Training Engine
Drives Applio's full pipeline: preprocess → extract → train.
All subprocess calls use cwd=APPLIO_ROOT (Applio's root directory).
Checkpoints saved to {APPLIO_ROOT}/logs/{model_name}/
"""

import os
import re
import subprocess
import threading
from pathlib import Path

# Applio is cloned at rvc/ in the project root
APPLIO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rvc")
)
# Applio saves everything under its own logs/
APPLIO_LOGS = os.path.join(APPLIO_ROOT, "logs")

PYTHON = "python"

# Regex matching Applio's stdout:
# "mymodel | epoch=42 | step=1250 | time=... | lowest_value=0.1234 ..."
EPOCH_RE = re.compile(r"epoch=(\d+)")
LOSS_RE  = re.compile(r"lowest_value=([0-9]+\.[0-9]+)")


class TrainingEngine:

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_training(
        self,
        dataset_path: str,
        model_name: str,
        hyperparams: dict,
        progress_callback=None,
    ) -> str:
        """
        Run the full Applio pipeline:
          1. preprocess  → slice audio into mel-spec training data
          2. extract     → F0 pitch + HuBERT embeddings
          3. train       → RVC model training (streams epoch/loss)

        Returns path to the final G_*.pth checkpoint.
        """
        dataset_path = os.path.abspath(dataset_path)
        batch_size   = hyperparams.get("batch_size", 4)
        epochs       = hyperparams.get("epochs", 300)
        save_every   = hyperparams.get("save_every_n_epochs", 50)
        cpu_cores    = os.cpu_count() or 4

        # ── Step 1: Preprocess ─────────────────────────────────────────
        self._run_step(
            label="Preprocess",
            cmd=[
                PYTHON,
                os.path.join("rvc", "train", "preprocess", "preprocess.py"),
                os.path.join(APPLIO_LOGS, model_name),  # output log dir
                dataset_path,                            # input audio dir
                "48000",                                 # sample rate
                str(cpu_cores),
                "Automatic",                             # cut_preprocess
                "False",                                 # process_effects
                "False",                                 # noise_reduction
                "0.7",                                   # clean_strength
                "3.0",                                   # chunk_len (seconds)
                "0.3",                                   # overlap_len
                "none",                                  # normalization_mode
            ],
        )

        # ── Step 2: Extract F0 + embeddings ───────────────────────────
        self._run_step(
            label="Extract",
            cmd=[
                PYTHON,
                os.path.join("rvc", "train", "extract", "extract.py"),
                os.path.join(APPLIO_LOGS, model_name),
                "rmvpe",        # f0_method
                str(cpu_cores),
                "0",            # gpu index
                "48000",        # sample_rate
                "contentvec",   # embedder_model
                "None",         # embedder_model_custom
                "2",            # include_mutes
            ],
        )

        # ── Step 3: Train (streams stdout) ────────────────────────────
        train_cmd = [
            PYTHON, "-u",
            os.path.join("rvc", "train", "train.py"),
            model_name,
            str(save_every),   # save_every_epoch
            str(epochs),       # total_epoch
            os.path.join("rvc", "models", "pretraineds", "refinegan", "f0G48k.pth"),
            os.path.join("rvc", "models", "pretraineds", "refinegan", "f0D48k.pth"),
            "0",               # gpus
            str(batch_size),
            "48000",           # sample_rate
            "true",            # save_only_latest
            "true",            # save_every_weights
            "false",           # cache_data_in_gpu
            "false",           # overtraining_detector
            "50",              # overtraining_threshold
            "false",           # cleanup
            "RefineGAN",       # vocoder
            "false",           # checkpointing
        ]

        print(f"[TrainingEngine] Starting training: {model_name}")
        with self._lock:
            self._process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=APPLIO_ROOT,
            )

        last_epoch = 0
        last_loss  = 0.0

        for line in self._process.stdout:
            line = line.rstrip()
            if line:
                print(f"[RVC] {line}")

            e_match = EPOCH_RE.search(line)
            l_match = LOSS_RE.search(line)

            if e_match:
                last_epoch = int(e_match.group(1))
            if l_match:
                last_loss = float(l_match.group(1))

            if e_match and l_match and progress_callback:
                try:
                    progress_callback(last_epoch, last_loss)
                except Exception as cb_err:
                    print(f"[TrainingEngine] Callback error: {cb_err}")

        self._process.wait()
        ret = self._process.returncode
        with self._lock:
            self._process = None

        if ret != 0:
            raise RuntimeError(f"Training exited with code {ret}")

        final_path = self._find_latest_checkpoint(model_name)
        print(f"[TrainingEngine] Done. Checkpoint: {final_path}")
        return final_path

    def stop_training(self) -> None:
        """Gracefully terminate the training subprocess."""
        with self._lock:
            proc = self._process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("[TrainingEngine] Training stopped.")
        else:
            print("[TrainingEngine] No active training process.")

    def get_checkpoint_list(self, model_name: str) -> list:
        """Return sorted list of G_*.pth checkpoints for model_name."""
        model_dir = os.path.join(APPLIO_LOGS, model_name)
        if not os.path.isdir(model_dir):
            return []
        return sorted(str(p) for p in Path(model_dir).glob("G_*.pth"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_step(self, label: str, cmd: list) -> None:
        """Run a blocking subprocess step from APPLIO_ROOT."""
        print(f"[TrainingEngine] {label}: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=APPLIO_ROOT,
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} step failed with exit code {result.returncode}"
            )
        print(f"[TrainingEngine] {label} complete.")

    def _find_latest_checkpoint(self, model_name: str) -> str:
        checkpoints = self.get_checkpoint_list(model_name)
        if checkpoints:
            return checkpoints[-1]
        return os.path.join(APPLIO_LOGS, model_name, f"G_2333333.pth")
