"""
Application Backend
Single entry-point that wires all modules together.
Designed for future UI / API layer consumption.
Every operation returns a structured result dict — no raw exceptions to callers.
"""

import os
import time
import threading
from datetime import datetime, timezone

from core.preprocessing   import AudioPreprocessor, PreprocessingError
from core.llm_config      import LLMConfigurator
from core.training_engine import TrainingEngine
from core.inference_engine import InferenceEngine
from services.session_manager import SessionManager
from services.supabase_client import SupabaseClient


class AppBackend:
    """
    Stateful application backend.
    One instance per application process; sessions survive across calls.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir     = os.path.abspath(base_dir)
        self.sessions     = SessionManager()
        self.supabase     = SupabaseClient()
        self._engines:  dict[str, TrainingEngine]   = {}   # model_name → engine
        self._threads:  dict[str, threading.Thread] = {}   # session_id → thread
        self._lock      = threading.Lock()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> dict:
        ok = self.supabase.login(email, password)
        return {"ok": ok, "user_id": self.supabase.user_id}

    def register(self, email: str, password: str) -> dict:
        ok = self.supabase.register(email, password)
        return {"ok": ok, "user_id": self.supabase.user_id}

    def logout(self) -> dict:
        self.supabase.logout()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Pipeline — blocking (for CLI) or async (for UI)
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        files:      list,
        model_name: str,
        hyperparams_override: dict | None = None,
        async_mode: bool = False,
    ) -> dict:
        """
        Start the full preprocessing → config → training pipeline.

        async_mode=True: returns immediately with session_id; training runs
                         in a background thread — poll get_session_status().
        async_mode=False: blocks until training completes.
        """
        session_id = self.sessions.create_session(
            model_name=model_name,
            files=files,
            user_id=self.supabase.user_id,
        )

        if async_mode:
            t = threading.Thread(
                target=self._pipeline_worker,
                args=(session_id, files, model_name, hyperparams_override),
                daemon=True,
            )
            with self._lock:
                self._threads[session_id] = t
            t.start()
            return {"ok": True, "session_id": session_id, "async": True}
        else:
            return self._pipeline_worker(session_id, files, model_name,
                                         hyperparams_override)

    def resume_pipeline(
        self,
        session_id:  str,
        async_mode:  bool = False,
    ) -> dict:
        """Resume a paused/stopped training session."""
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": f"Session {session_id} not found"}

        model_name  = session["model_name"]
        files       = session["files"]
        hyperparams = session["hyperparams"]

        if async_mode:
            t = threading.Thread(
                target=self._pipeline_worker,
                args=(session_id, files, model_name, hyperparams, True),
                daemon=True,
            )
            with self._lock:
                self._threads[session_id] = t
            t.start()
            return {"ok": True, "session_id": session_id, "async": True}

        return self._pipeline_worker(session_id, files, model_name,
                                     hyperparams, resume=True)

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def stop_training(self, session_id: str, force: bool = False) -> dict:
        """Stop training for a session. Checkpoint is preserved."""
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        model_name = session["model_name"]
        with self._lock:
            engine = self._engines.get(model_name)
        if not engine:
            return {"ok": False, "error": "No active training engine"}
        if force:
            engine.force_stop()
        else:
            engine.stop_training()
        self.sessions.set_status(session_id, "paused")
        checkpoint = engine.get_checkpoint_list(model_name)
        latest = checkpoint[-1] if checkpoint else None
        if latest:
            self.sessions.set_checkpoint(session_id, latest)
        return {"ok": True, "checkpoint": latest}

    def get_session_status(self, session_id: str) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        model_name = session["model_name"]
        with self._lock:
            engine = self._engines.get(model_name)
        engine_status = engine.get_training_status(model_name) if engine else {}
        return {"ok": True, "session": session, "engine": engine_status}

    def list_sessions(self, user_id: str | None = None) -> dict:
        return {"ok": True, "sessions": self.sessions.list_sessions(user_id)}

    # ------------------------------------------------------------------
    # Segment inspection
    # ------------------------------------------------------------------

    def get_segments(self, session_id: str) -> dict:
        """Return all segments from the manifest for user review."""
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        preprocessor = AudioPreprocessor(base_dir=self.base_dir)
        return {"ok": True, "segments": preprocessor.get_segments()}

    def approve_segments(self, session_id: str, approved_paths: list) -> dict:
        """Mark specific segments as approved for training."""
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        preprocessor = AudioPreprocessor(base_dir=self.base_dir)
        count = preprocessor.approve_segments(approved_paths)
        self.sessions.update_session(
            session_id,
            segment_manifest=preprocessor.manifest,
        )
        return {"ok": True, "approved_count": count}

    def delete_segment(self, session_id: str, path: str) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        preprocessor = AudioPreprocessor(base_dir=self.base_dir)
        ok = preprocessor.delete_segment(path)
        return {"ok": ok}

    # ------------------------------------------------------------------
    # Checkpoint operations
    # ------------------------------------------------------------------

    def list_checkpoints(self, session_id: str) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        engine = TrainingEngine()
        checkpoints = engine.get_checkpoint_list(session["model_name"])
        return {"ok": True, "checkpoints": checkpoints}

    def test_checkpoint(
        self,
        session_id:       str,
        checkpoint_path:  str,
        test_audio_path:  str,
        pitch_shift:      int = 0,
    ) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        try:
            engine = TrainingEngine()
            output = engine.test_checkpoint(
                model_name=session["model_name"],
                checkpoint_path=checkpoint_path,
                test_audio_path=test_audio_path,
                pitch_shift=pitch_shift,
            )
            return {"ok": True, "output_path": output}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def convert_audio(
        self,
        model_path:        str,
        input_audio_path:  str,
        output_path:       str,
        pitch_shift:       int = 0,
    ) -> dict:
        try:
            engine = InferenceEngine()
            out    = engine.convert(model_path, input_audio_path,
                                    output_path, pitch_shift)
            return {"ok": True, "output_path": out}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_models(self) -> dict:
        engine = InferenceEngine()
        return {"ok": True, "models": engine.list_models()}

    # ------------------------------------------------------------------
    # Pipeline worker (runs in thread or inline)
    # ------------------------------------------------------------------

    def _pipeline_worker(
        self,
        session_id:           str,
        files:                list,
        model_name:           str,
        hyperparams_override: dict | None = None,
        resume:               bool = False,
    ) -> dict:
        start = time.time()
        try:
            # ── Preprocess ────────────────────────────────────────────
            self.sessions.set_status(session_id, "preprocessing")
            preprocessor = AudioPreprocessor(base_dir=self.base_dir)

            if resume:
                # Use already-approved segments from manifest
                approved = preprocessor.get_approved_segment_paths()
                if not approved:
                    # Re-run preprocessing from scratch
                    stats = preprocessor.process(files)
                    preprocessor.approve_all()
                else:
                    import wave
                    total_s = sum(
                        _wav_duration(p) for p in approved if os.path.exists(p)
                    )
                    stats = {
                        "segment_paths":          approved,
                        "total_duration_minutes": round(total_s / 60, 2),
                        "segment_count":          len(approved),
                        "quality_label":          _quality(total_s / 60),
                        "manifest_path":          preprocessor.manifest,
                    }
            else:
                stats = preprocessor.process(files)
                preprocessor.approve_all()

            self.sessions.update_session(
                session_id,
                segment_manifest=preprocessor.manifest,
                status="preprocessing_done",
            )

            # ── Hyperparameters ────────────────────────────────────────
            if hyperparams_override:
                hyperparams = hyperparams_override
            else:
                configurator = LLMConfigurator()
                hyperparams  = configurator.get_hyperparameters(stats)

            self.sessions.update_session(session_id, hyperparams=hyperparams,
                                         total_epochs=hyperparams.get("epochs", 0))

            # ── Training ──────────────────────────────────────────────
            self.sessions.set_status(session_id, "training")
            engine = TrainingEngine()
            with self._lock:
                self._engines[model_name] = engine

            def on_progress(epoch: int, loss: float):
                self.sessions.update_progress(session_id, epoch, loss)

            model_path = engine.start_training(
                dataset_path=os.path.join(self.base_dir, "dataset", "segments"),
                model_name=model_name,
                hyperparams=hyperparams,
                progress_callback=on_progress,
                resume=resume,
            )

            self.sessions.set_status(session_id, "done")
            self.sessions.set_checkpoint(session_id, model_path)

            elapsed = int(time.time() - start)

            # ── Optional Supabase sync ─────────────────────────────────
            if self.supabase.is_authenticated:
                self.supabase.sync_training_session({
                    "user_id":          self.supabase.user_id,
                    "model_name":       model_name,
                    "status":           "completed",
                    "duration_seconds": elapsed,
                    "epochs_completed": self.sessions.get_session(session_id).get("current_epoch", 0),
                    "hardware_profile": LLMConfigurator()._scan_hardware(),
                    "session_id":       session_id,
                    "checkpoint_path":  model_path,
                    "created_at":       datetime.now(timezone.utc).isoformat(),
                })

            return {"ok": True, "model_path": model_path,
                    "session_id": session_id, "elapsed_s": elapsed}

        except PreprocessingError as e:
            self.sessions.set_status(session_id, "error", str(e))
            return {"ok": False, "error": str(e), "session_id": session_id}
        except Exception as e:
            self.sessions.set_status(session_id, "error", str(e))
            return {"ok": False, "error": str(e), "session_id": session_id}
        finally:
            with self._lock:
                self._engines.pop(model_name, None)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _wav_duration(path: str) -> float:
    import wave
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _quality(minutes: float) -> str:
    if minutes < 10:
        return "Poor"
    if minutes <= 20:
        return "Acceptable"
    return "Optimal"
