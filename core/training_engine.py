"""
RVC Training Engine
Drives Applio's full pipeline: preprocess → extract → train.
Supports:
  - Resume from existing checkpoint (skips preprocess/extract if done)
  - Per-epoch checkpoint saving
  - Graceful stop with guaranteed latest checkpoint saved
  - Checkpoint testing via inference
"""

import os
import re
import signal
import subprocess
import threading
from pathlib import Path

APPLIO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rvc")
)
APPLIO_LOGS = os.path.join(APPLIO_ROOT, "logs")
PYTHON      = "python"

EPOCH_RE = re.compile(r"epoch=(\d+)")
LOSS_RE  = re.compile(r"lowest_value=([0-9]+\.[0-9]+)")
STEP_RE  = re.compile(r"step=(\d+)")


class TrainingEngine:

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._lock    = threading.Lock()
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_training(
        self,
        dataset_path: str,
        model_name:   str,
        hyperparams:  dict,
        progress_callback=None,
        resume: bool = False,
    ) -> str:
        """
        Run the full Applio pipeline.

        Args:
            resume: If True, skip preprocess+extract when they've already run.
                    Training auto-resumes from the latest checkpoint.
        Returns:
            Path to the latest G_*.pth checkpoint.
        """
        self._stop_requested = False
        dataset_path = os.path.abspath(dataset_path)
        batch_size   = hyperparams.get("batch_size", 4)
        epochs       = hyperparams.get("epochs", 300)
        save_every   = hyperparams.get("save_every_n_epochs", 50)
        cpu_cores    = os.cpu_count() or 4

        segments_ready    = self._segments_ready(dataset_path)
        rvc_preprocessed  = self._rvc_preprocess_done(model_name)
        already_extracted = self._extraction_done(model_name)
        latest_ckpt       = self._find_latest_g_checkpoint(model_name)
        has_checkpoint    = latest_ckpt is not None

        if has_checkpoint:
            print(f"[Training] Resuming from checkpoint {os.path.basename(latest_ckpt)}")

        if resume or has_checkpoint:
            print("[Pipeline] Skipping preprocess (resume mode)")
        elif segments_ready and rvc_preprocessed:
            print("[Pipeline] Skipping preprocess (already exists)")
        else:
            self._run_step(
                label="Preprocess",
                cmd=[
                    PYTHON,
                    os.path.join("rvc", "train", "preprocess", "preprocess.py"),
                    os.path.join(APPLIO_LOGS, model_name),
                    dataset_path,
                    "48000",
                    str(cpu_cores),
                    "Automatic",
                    "False",
                    "False",
                    "0.7",
                    "3.0",
                    "0.3",
                    "none",
                ],
            )

        if already_extracted:
            if resume or has_checkpoint:
                print("[Pipeline] Skipping extract (resume mode)")
            else:
                print("[Pipeline] Skipping extract (already exists)")
        else:
            self._run_step(
                label="Extract",
                cmd=[
                    PYTHON,
                    os.path.join("rvc", "train", "extract", "extract.py"),
                    os.path.join(APPLIO_LOGS, model_name),
                    "rmvpe",
                    str(cpu_cores),
                    "0",
                    "48000",
                    "contentvec",
                    "None",
                    "2",
                ],
            )

        # Train — Applio auto-resumes from latest G_*.pth if it exists
        train_cmd = [
            PYTHON, "-u",
            os.path.join("rvc", "train", "train.py"),
            model_name,
            str(save_every),
            str(epochs),
            os.path.join("rvc", "models", "pretraineds", "hifi-gan", "f0G48k.pth"),
            os.path.join("rvc", "models", "pretraineds", "hifi-gan", "f0D48k.pth"),
            "0",
            str(batch_size),
            "48000",
            "true",   # save_only_latest
            "true",   # save_every_weights
            "false",  # cache_data_in_gpu
            "false",  # overtraining_detector
            "50",     # overtraining_threshold
            "false",  # cleanup
            "HiFi-GAN",
            "false",  # checkpointing
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

        try:
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

                # Honour stop request between epochs
                if self._stop_requested and e_match:
                    print("[TrainingEngine] Stop requested — terminating after epoch.")
                    with self._lock:
                        if self._process and self._process.poll() is None:
                            self._process.terminate()
                    break

            self._process.wait()
        finally:
            with self._lock:
                if self._process and self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()

        ret = self._process.returncode if self._process else -1
        with self._lock:
            self._process = None

        if ret not in (0, -15):  # -15 = SIGTERM (graceful stop)
            raise RuntimeError(f"Training exited with code {ret}")

        final_path = self._find_latest_checkpoint(model_name)
        if not os.path.exists(final_path):
            raise RuntimeError(f"Training finished but checkpoint was not created: {final_path}")

        print(f"[TrainingEngine] Done. Checkpoint: {final_path}")
        return final_path

    def resume_training(
        self,
        dataset_path: str,
        model_name:   str,
        hyperparams:  dict,
        progress_callback=None,
    ) -> str:
        """Convenience wrapper — always sets resume=True."""
        return self.start_training(
            dataset_path=dataset_path,
            model_name=model_name,
            hyperparams=hyperparams,
            progress_callback=progress_callback,
            resume=True,
        )

    def stop_training(self) -> None:
        """Signal training to stop cleanly after the current epoch completes."""
        self._stop_requested = True
        with self._lock:
            proc = self._process
        if proc and proc.poll() is None:
            print("[TrainingEngine] Stop signal sent — waiting for epoch boundary...")
        else:
            print("[TrainingEngine] No active training process.")

    def force_stop(self) -> None:
        """Immediately terminate training (checkpoint may be incomplete)."""
        with self._lock:
            proc = self._process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("[TrainingEngine] Training force-stopped.")
        with self._lock:
            self._process = None

    def get_checkpoint_list(self, model_name: str) -> list:
        """Return sorted list of per-epoch *_Ne_*s.pth checkpoints."""
        model_dir = os.path.join(APPLIO_LOGS, model_name)
        if not os.path.isdir(model_dir):
            return []
        # Per-epoch named checkpoints: e.g. mymodel_50e_1100s.pth
        named = sorted(
            str(p) for p in Path(model_dir).glob(f"{model_name}_*e_*s.pth")
        )
        # Latest-only checkpoint
        latest = str(Path(model_dir) / "G_2333333.pth")
        result = named
        if os.path.exists(latest) and latest not in result:
            result = result + [latest]
        return result

    def test_checkpoint(
        self,
        model_name:      str,
        checkpoint_path: str,
        test_audio_path: str,
        output_path:     str | None = None,
        pitch_shift:     int = 0,
    ) -> str:
        """
        Run inference on a checkpoint to listen to its quality.
        Returns path to the converted output WAV.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(test_audio_path):
            raise FileNotFoundError(f"Test audio not found: {test_audio_path}")

        if output_path is None:
            ckpt_stem   = Path(checkpoint_path).stem
            output_path = os.path.join(
                APPLIO_LOGS, model_name, f"test_{ckpt_stem}.wav"
            )

        from core.inference_engine import InferenceEngine
        engine = InferenceEngine()
        result = engine.convert(
            model_path=checkpoint_path,
            input_audio_path=test_audio_path,
            output_path=output_path,
            pitch_shift=pitch_shift,
        )
        print(f"[TrainingEngine] Checkpoint test saved: {result}")
        return result

    def get_training_status(self, model_name: str) -> dict:
        """Return current training state for a model."""
        checkpoints   = self.get_checkpoint_list(model_name)
        is_running    = False
        with self._lock:
            is_running = self._process is not None and self._process.poll() is None

        latest_epoch  = 0
        if checkpoints:
            # Parse epoch from filename like mymodel_42e_924s.pth
            m = re.search(r"_(\d+)e_", Path(checkpoints[-1]).name)
            if m:
                latest_epoch = int(m.group(1))

        return {
            "model_name":    model_name,
            "is_running":    is_running,
            "checkpoints":   checkpoints,
            "latest_epoch":  latest_epoch,
            "extraction_done": self._extraction_done(model_name),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extraction_done(self, model_name: str) -> bool:
        """True if filelist.txt exists and extracted feature files are present."""
        model_dir = Path(APPLIO_LOGS) / model_name
        filelist = model_dir / "filelist.txt"
        if not filelist.exists() or filelist.stat().st_size <= 0:
            return False
        try:
            next(model_dir.rglob("*.npy"))
            return True
        except StopIteration:
            return False

    def _run_step(self, label: str, cmd: list) -> None:
        print(f"[TrainingEngine] {label}: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=APPLIO_ROOT, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"{label} step failed with exit code {result.returncode}")
        print(f"[TrainingEngine] {label} complete.")

    def _find_latest_checkpoint(self, model_name: str) -> str:
        checkpoints = self.get_checkpoint_list(model_name)
        if checkpoints:
            return checkpoints[-1]
        return os.path.join(APPLIO_LOGS, model_name, "G_2333333.pth")

    def _find_latest_g_checkpoint(self, model_name: str) -> str | None:
        model_dir = Path(APPLIO_LOGS) / model_name
        if not model_dir.is_dir():
            return None
        g_files = sorted(
            model_dir.glob("G_*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
        if not g_files:
            return None
        return str(g_files[-1])

    def _segments_ready(self, dataset_path: str) -> bool:
        segments_dir = dataset_path
        if os.path.basename(segments_dir) != "segments":
            segments_dir = os.path.join(segments_dir, "segments")
        if not os.path.isdir(segments_dir):
            return False
        try:
            return any(entry.is_file() for entry in os.scandir(segments_dir))
        except OSError:
            return False

    def _rvc_preprocess_done(self, model_name: str) -> bool:
        model_dir = Path(APPLIO_LOGS) / model_name
        sliced_dir = model_dir / "sliced_audios"
        if not sliced_dir.is_dir():
            return False
        try:
            return any(p.suffix == ".wav" for p in sliced_dir.iterdir())
        except OSError:
            return False
