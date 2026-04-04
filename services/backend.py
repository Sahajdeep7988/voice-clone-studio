"""
Application Backend
Single entry-point wiring all modules together.
Every public method returns a structured result dict — no raw exceptions.
"""

import os
import time
import threading

from core.preprocessing    import AudioPreprocessor, PreprocessingError
from core.llm_config       import LLMConfigurator
from core.training_engine  import TrainingEngine
from core.inference_engine import InferenceEngine
from core.logger           import get_logger
from services.session_manager  import SessionManager
from services.supabase_client  import SupabaseClient

# Minimum segments required before training is allowed
MIN_TRAINING_SEGMENTS = 10


class AppBackend:
    """
    Stateful application backend.
    One instance per process; sessions survive across calls.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir  = os.path.abspath(base_dir)
        self.sessions  = SessionManager()
        self.supabase  = SupabaseClient()
        self._engines: dict[str, TrainingEngine]   = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock     = threading.Lock()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> dict:
        return self.supabase.login(email, password)

    def register(self, email: str, password: str) -> dict:
        return self.supabase.register(email, password)

    def logout(self) -> dict:
        self.supabase.logout()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        files:                list,
        model_name:           str,
        hyperparams_override: dict | None = None,
        async_mode:           bool = False,
        max_workers:          int  = 4,
        resume:              bool | None = None,
        user_id:             str | None = None,
        access_token:        str | None = None,
    ) -> dict:
        """
        Start preprocessing -> config -> training.

        async_mode=True: returns immediately; poll get_session_status().
        async_mode=False: blocks until training completes.
        max_workers: parallel preprocessing workers.
        """
        session_id = None
        if resume is None:
            checkpoint_exists = len(TrainingEngine().get_checkpoint_list(model_name)) > 0
            resume = checkpoint_exists

        if session_id is None:
            session_id = self.sessions.create_session(
                model_name=model_name,
                files=files,
                user_id=user_id,
            )

        if async_mode:
            t = threading.Thread(
                target=self._pipeline_worker,
                args=(session_id, files, model_name, hyperparams_override, resume, max_workers, user_id, access_token),
                daemon=True,
            )
            with self._lock:
                self._threads[session_id] = t
            t.start()
            return {"ok": True, "session_id": session_id, "async": True}

        return self._pipeline_worker(
            session_id, files, model_name, hyperparams_override,
            resume=resume, max_workers=max_workers, user_id=user_id, access_token=access_token,
        )

    def resume_pipeline(
        self,
        session_id: str,
        async_mode: bool = False,
        access_token: str | None = None,
    ) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": f"Session {session_id} not found"}

        model_name  = session["model_name"]
        files       = session["files"]
        hyperparams = session["hyperparams"]

        if async_mode:
            t = threading.Thread(
                target=self._pipeline_worker,
                args=(session_id, files, model_name, hyperparams, True, 4, session.get("user_id"), access_token),
                daemon=True,
            )
            with self._lock:
                self._threads[session_id] = t
            t.start()
            return {"ok": True, "session_id": session_id, "async": True}

        return self._pipeline_worker(
            session_id, files, model_name, hyperparams, resume=True,
            user_id=session.get("user_id"), access_token=access_token,
        )

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def stop_training(
        self,
        session_id:   str,
        force:        bool = False,
        user_id:      str | None = None,
        access_token: str | None = None,
    ) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        with self._lock:
            engine = self._engines.get(session_id)
        if not engine:
            return {"ok": False, "error": "No active training engine"}
        if force:
            engine.force_stop()
        else:
            engine.stop_training()
        self.sessions.set_status(session_id, "paused")
        checkpoint = engine.get_checkpoint_list(session["model_name"])
        latest     = checkpoint[-1] if checkpoint else None
        if latest:
            self.sessions.set_checkpoint(session_id, latest)
        self._sync_supabase(session_id, user_id, access_token,
                            status="paused", checkpoint_path=latest)
        return {"ok": True, "checkpoint": latest}

    def get_session_status(self, session_id: str) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        with self._lock:
            engine = self._engines.get(session_id)
        engine_status = engine.get_training_status(session["model_name"]) if engine else {}
        return {"ok": True, "session": session, "engine": engine_status}

    def list_sessions(self, user_id: str | None = None) -> dict:
        return {"ok": True, "sessions": self.sessions.list_sessions(user_id)}

    # ------------------------------------------------------------------
    # Segment inspection
    # ------------------------------------------------------------------

    def get_segments(self, session_id: str) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        session_base = os.path.join(self.base_dir, "sessions_data", session_id)
        preprocessor = AudioPreprocessor(base_dir=session_base)
        return {"ok": True, "segments": preprocessor.get_segments()}

    def approve_segments(self, session_id: str, approved_paths: list) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        session_base = os.path.join(self.base_dir, "sessions_data", session_id)
        preprocessor = AudioPreprocessor(base_dir=session_base)
        count = preprocessor.approve_segments(approved_paths)
        self.sessions.update_session(session_id, segment_manifest=preprocessor.manifest)
        return {"ok": True, "approved_count": count}

    def delete_segment(self, session_id: str, path: str) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        session_base = os.path.join(self.base_dir, "sessions_data", session_id)
        preprocessor = AudioPreprocessor(base_dir=session_base)
        ok = preprocessor.delete_segment(path)
        return {"ok": ok}

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self, session_id: str) -> dict:
        session = self.sessions.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}
        engine      = TrainingEngine()
        checkpoints = engine.get_checkpoint_list(session["model_name"])
        return {"ok": True, "checkpoints": checkpoints}

    def test_checkpoint(
        self,
        session_id:      str,
        checkpoint_path: str,
        test_audio_path: str,
        pitch_shift:     int = 0,
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
        model_path:       str,
        input_audio_path: str,
        output_path:      str,
        pitch_shift:      int = 0,
    ) -> dict:
        try:
            engine = InferenceEngine()
            out    = engine.convert(model_path, input_audio_path, output_path, pitch_shift)
            return {"ok": True, "output_path": out}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_models(self) -> dict:
        engine = InferenceEngine()
        return {"ok": True, "models": engine.list_models()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataset(self, dataset_path: str) -> dict:
        """Verify dataset is ready for training."""
        log = get_logger(stage="validate")

        if not os.path.isdir(dataset_path):
            return {"ok": False, "error": f"Dataset path not found: {dataset_path}"}

        all_wavs = [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith(".wav")
        ]

        import wave

        # Remove empty files and count valid ones
        valid  = []
        empty  = []
        for f in all_wavs:
            full = os.path.join(dataset_path, f)
            if os.path.getsize(full) > 0:
                try:
                    with wave.open(full, "rb") as wf:
                        if wf.getnframes() > 0:
                            valid.append(f)
                        else:
                            empty.append(full)
                except Exception:
                    empty.append(full)
            else:
                empty.append(full)

        if empty:
            log.warning(f"Removing {len(empty)} empty segment files")
            for path in empty:
                try:
                    os.remove(path)
                except OSError:
                    pass

        if not valid:
            return {"ok": False, "error": "No audio segments found in dataset"}

        if len(valid) < MIN_TRAINING_SEGMENTS:
            return {
                "ok": False,
                "error": (
                    f"Too few segments: {len(valid)} "
                    f"(minimum {MIN_TRAINING_SEGMENTS} required)"
                ),
            }

        log.success(f"Dataset valid: {len(valid)} segments at {dataset_path}")
        return {"ok": True, "segment_count": len(valid)}

    # ------------------------------------------------------------------
    # Supabase sync helper
    # ------------------------------------------------------------------

    def _sync_supabase(
        self,
        session_id:   str,
        user_id:      str | None,
        access_token: str | None,
        **extra,
    ) -> None:
        """
        Fire-and-forget Supabase upsert.  Silently skips when user_id or
        access_token are absent (e.g. CLI run without --email/--password,
        or unauthenticated local use).
        """
        if not user_id or not access_token:
            return
        session = self.sessions.get_session(session_id)
        if not session:
            return
        record = {
            "session_id": session_id,
            "user_id":    user_id,
            "model_name": session.get("model_name", ""),
            "status":     session.get("status", ""),
            **extra,
        }
        self.supabase.sync_training_session(record, access_token=access_token)

    # ------------------------------------------------------------------
    # Pipeline worker
    # ------------------------------------------------------------------

    def _pipeline_worker(
        self,
        session_id:           str,
        files:                list,
        model_name:           str,
        hyperparams_override: dict | None = None,
        resume:               bool        = False,
        max_workers:          int         = 4,
        user_id:              str | None = None,
        access_token:         str | None = None,
    ) -> dict:
        log   = get_logger(stage="pipeline")
        start = time.time()
        checkpoint_exists = len(TrainingEngine().get_checkpoint_list(model_name)) > 0
        resume = resume or checkpoint_exists
        print(f"[Pipeline] Resume mode: {resume}")

        try:
            # ── Sync: session created ─────────────────────────────────
            self._sync_supabase(session_id, user_id, access_token,
                                status="preprocessing",
                                files=files)

            # ── Preprocess ────────────────────────────────────────────
            self.sessions.set_status(session_id, "preprocessing")
            session_base = os.path.join(self.base_dir, "sessions_data", session_id)
            preprocessor = AudioPreprocessor(base_dir=session_base)

            with log.timed("preprocess"):
                if resume:
                    approved = preprocessor.get_approved_segment_paths()
                    if not approved:
                        stats = preprocessor.process(files, max_workers=max_workers)
                        preprocessor.approve_all()
                    else:
                        print("[Pipeline] Skipping preprocess (resume mode)")
                        import wave as _wave

                        def _dur(p):
                            try:
                                with _wave.open(p, "rb") as wf:
                                    return wf.getnframes() / wf.getframerate()
                            except Exception:
                                return 0.0

                        total_s = sum(_dur(p) for p in approved if os.path.exists(p))
                        stats = {
                            "segment_paths":          approved,
                            "total_duration_minutes": round(total_s / 60, 2),
                            "segment_count":          len(approved),
                            "quality_label":          _quality(total_s / 60),
                            "manifest_path":          preprocessor.manifest,
                            "total_files":            len(files),
                            "success_count":          len(files),
                            "failed_count":           0,
                            "failures":               [],
                        }
                else:
                    stats = preprocessor.process(files, max_workers=max_workers)
                    preprocessor.approve_all()

            self.sessions.update_session(
                session_id,
                segment_manifest=preprocessor.manifest,
                status="preprocessing_done",
            )

            # ── Sync: preprocessing done ──────────────────────────────
            self._sync_supabase(session_id, user_id, access_token,
                                status="preprocessing_done",
                                segment_manifest=preprocessor.manifest)

            if stats.get("failed_count", 0) > 0:
                log.warning(
                    f"Preprocessing partial: "
                    f"{stats['success_count']}/{stats['total_files']} files OK, "
                    f"failures: {stats['failures']}"
                )

            # ── Validate dataset ──────────────────────────────────────
            dataset_path = os.path.join(session_base, "dataset", "segments")
            validation   = self._validate_dataset(dataset_path)
            if not validation["ok"]:
                self.sessions.set_status(session_id, "error", validation["error"])
                self._sync_supabase(session_id, user_id, access_token,
                                    status="error",
                                    error_message=validation["error"])
                return {"ok": False, "error": validation["error"], "session_id": session_id}

            # ── Hyperparameters ───────────────────────────────────────
            if hyperparams_override:
                hyperparams = hyperparams_override
            else:
                with log.timed("llm_config"):
                    configurator = LLMConfigurator()
                    hyperparams  = configurator.get_hyperparameters(stats)

            self.sessions.update_session(
                session_id,
                hyperparams=hyperparams,
                total_epochs=hyperparams.get("epochs", 0),
            )

            # ── Train ─────────────────────────────────────────────────
            self.sessions.set_status(session_id, "training")
            engine = TrainingEngine()
            with self._lock:
                self._engines[session_id] = engine

            # ── Sync: training started ────────────────────────────────
            self._sync_supabase(session_id, user_id, access_token,
                                status="training",
                                hyperparams=hyperparams,
                                total_epochs=hyperparams.get("epochs", 0),
                                hardware_profile=LLMConfigurator()._scan_hardware())

            # Throttle progress syncs: at most one every 30 seconds.
            _last_sync = [0.0]

            def on_progress(epoch: int, loss: float):
                self.sessions.update_progress(session_id, epoch, loss)
                get_logger(model_name, "train").info(f"epoch={epoch} loss={loss:.4f}")
                now = time.time()
                if now - _last_sync[0] >= 30:
                    _last_sync[0] = now
                    self._sync_supabase(session_id, user_id, access_token,
                                        status="training",
                                        current_epoch=epoch,
                                        latest_loss=loss,
                                        total_epochs=hyperparams.get("epochs", 0))

            with log.timed("training"):
                model_path = engine.start_training(
                    dataset_path=os.path.join(session_base, "dataset", "segments"),
                    model_name=model_name,
                    hyperparams=hyperparams,
                    progress_callback=on_progress,
                    resume=resume,
                )

            self.sessions.set_status(session_id, "done")
            self.sessions.set_checkpoint(session_id, model_path)

            elapsed          = int(time.time() - start)
            final_session    = self.sessions.get_session(session_id)
            epochs_completed = final_session.get("current_epoch", 0) if final_session else 0

            # ── Sync: completed ───────────────────────────────────────
            self._sync_supabase(session_id, user_id, access_token,
                                status="completed",
                                checkpoint_path=model_path,
                                duration_seconds=elapsed,
                                epochs_completed=epochs_completed,
                                current_epoch=epochs_completed,
                                hardware_profile=LLMConfigurator()._scan_hardware())

            log.success(f"Pipeline done: model_path={model_path} elapsed={elapsed}s")
            return {
                "ok":         True,
                "model_path": model_path,
                "session_id": session_id,
                "elapsed_s":  elapsed,
                "preprocess": {
                    "total_files":   stats.get("total_files", len(files)),
                    "success_count": stats.get("success_count", len(files)),
                    "failed_count":  stats.get("failed_count", 0),
                    "failures":      stats.get("failures", []),
                },
            }

        except PreprocessingError as e:
            self.sessions.set_status(session_id, "error", str(e))
            get_logger(stage="pipeline").error(str(e))
            self._sync_supabase(session_id, user_id, access_token,
                                status="error", error_message=str(e))
            return {"ok": False, "error": str(e), "session_id": session_id}
        except Exception as e:
            self.sessions.set_status(session_id, "error", str(e))
            get_logger(stage="pipeline").error(str(e))
            self._sync_supabase(session_id, user_id, access_token,
                                status="error", error_message=str(e))
            return {"ok": False, "error": str(e), "session_id": session_id}
        finally:
            with self._lock:
                self._engines.pop(session_id, None)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _quality(minutes: float) -> str:
    if minutes < 10:
        return "Poor"
    if minutes <= 20:
        return "Acceptable"
    return "Optimal"
