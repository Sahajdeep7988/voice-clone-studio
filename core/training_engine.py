"""
RVC Training Engine
Wraps Applio's train.py as a managed subprocess.
Parses epoch/loss from stdout in real time.
Saves checkpoints to ~/VoiceClone/models/{model_name}/
"""

import os
import re
import subprocess
import threading
from pathlib import Path


# Path to the RVC train.py — resolved relative to this file
# In production this should point to your Applio clone or installed rvc package
RVC_TRAIN_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "rvc", "train", "train.py"
)

# Pretrained base models — adjust to your actual paths
PRETRAIN_G = os.path.expanduser("~/VoiceClone/pretrained/f0G48k.pth")
PRETRAIN_D = os.path.expanduser("~/VoiceClone/pretrained/f0D48k.pth")

MODELS_BASE = os.path.expanduser("~/VoiceClone/models")

# Regex patterns matching Applio's stdout format:
# e.g. "mymodel | epoch=42 | step=1250 | ... | lowest_value=0.1234 ..."
EPOCH_PATTERN = re.compile(r"epoch=(\d+)")
LOSS_PATTERN = re.compile(r"lowest_value=([0-9]+\.[0-9]+)")


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
        Launch RVC training as a subprocess, stream output, and return the
        final checkpoint path when done.

        Args:
            dataset_path: Path containing preprocessed .wav segments.
            model_name:   Identifier used for checkpoints and logs.
            hyperparams:  Dict with batch_size, epochs, learning_rate,
                          save_every_n_epochs.
            progress_callback: Callable(epoch: int, loss: float) called on
                          every parsed epoch line.

        Returns:
            Path to the final .pth checkpoint file.
        """
        model_dir = os.path.join(MODELS_BASE, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # First run RVC preprocessing on the dataset
        self._run_preprocessing(dataset_path, model_name, hyperparams)

        # Build the training command (matches Applio's sys.argv parsing)
        cmd = self._build_train_command(model_name, hyperparams)
        print(f"[TrainingEngine] Launching: {' '.join(cmd)}")

        with self._lock:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

        last_epoch = 0
        last_loss = 0.0

        for line in self._process.stdout:
            line = line.rstrip()
            if line:
                print(f"[RVC] {line}")

            epoch_match = EPOCH_PATTERN.search(line)
            loss_match = LOSS_PATTERN.search(line)

            if epoch_match:
                last_epoch = int(epoch_match.group(1))
            if loss_match:
                last_loss = float(loss_match.group(1))

            if epoch_match and loss_match and progress_callback:
                try:
                    progress_callback(last_epoch, last_loss)
                except Exception as cb_err:
                    print(f"[TrainingEngine] Callback error: {cb_err}")

        self._process.wait()
        ret = self._process.returncode

        with self._lock:
            self._process = None

        if ret != 0:
            raise RuntimeError(
                f"Training subprocess exited with code {ret}"
            )

        final_path = self._find_latest_checkpoint(model_name)
        print(f"[TrainingEngine] Training complete. Model: {final_path}")
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
        """Return all .pth checkpoint paths for model_name, sorted by name."""
        model_dir = os.path.join(MODELS_BASE, model_name)
        if not os.path.isdir(model_dir):
            return []
        return sorted(
            str(p) for p in Path(model_dir).glob("*.pth")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_preprocessing(
        self, dataset_path: str, model_name: str, hyperparams: dict
    ) -> None:
        """
        Run RVC's preprocessing step on the wav segments before training.
        Uses Applio-compatible preprocess.py call pattern.
        """
        preprocess_script = os.path.join(
            os.path.dirname(__file__), "..", "rvc", "train", "preprocess",
            "preprocess.py"
        )
        if not os.path.exists(preprocess_script):
            print(
                "[TrainingEngine] preprocess.py not found — skipping "
                "preprocessing step (assumes data is already prepared)."
            )
            return

        cmd = [
            "python", preprocess_script,
            os.path.abspath(dataset_path),
            "48000",           # sample rate
            str(os.cpu_count() or 4),
            model_name,
        ]
        print(f"[TrainingEngine] Preprocessing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"[TrainingEngine] Preprocessing warning: {result.stderr}"
            )
        else:
            print("[TrainingEngine] Preprocessing done.")

    def _build_train_command(self, model_name: str, hyperparams: dict) -> list:
        """
        Build argv list for Applio's train.py.
        16 positional arguments (index 1-16 in sys.argv).
        """
        batch_size = hyperparams.get("batch_size", 4)
        epochs = hyperparams.get("epochs", 300)
        save_every = hyperparams.get("save_every_n_epochs", 50)

        train_script = os.path.abspath(RVC_TRAIN_SCRIPT)

        return [
            "python", train_script,
            model_name,           # 1  model_name
            str(save_every),      # 2  save_every_epoch
            str(epochs),          # 3  total_epoch
            PRETRAIN_G,           # 4  pretrainG
            PRETRAIN_D,           # 5  pretrainD
            "0",                  # 6  gpus (GPU 0; use "0-1" for multi-GPU)
            str(batch_size),      # 7  batch_size
            "48000",              # 8  sample_rate
            "true",               # 9  save_only_latest
            "true",               # 10 save_every_weights
            "false",              # 11 cache_data_in_gpu
            "false",              # 12 overtraining_detector
            "50",                 # 13 overtraining_threshold
            "false",              # 14 cleanup
            "RefineGAN",          # 15 vocoder
            "false",              # 16 checkpointing
        ]

    def _find_latest_checkpoint(self, model_name: str) -> str:
        checkpoints = self.get_checkpoint_list(model_name)
        if not checkpoints:
            # Fallback path if checkpoints are stored elsewhere
            fallback = os.path.join(MODELS_BASE, model_name, f"{model_name}.pth")
            return fallback
        # Return the lexicographically last (highest epoch) checkpoint
        return checkpoints[-1]
