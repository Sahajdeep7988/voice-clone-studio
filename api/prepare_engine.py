"""
Two-step pipeline engine.
Exposes prepare() and confirm_and_stream() without touching any existing files.
"""

import asyncio
import json
import os
import sys
import wave as _wave

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.preprocessing import AudioPreprocessor
from core.llm_config import LLMConfigurator
from core.training_engine import TrainingEngine
from core.logger import get_logger
from services.backend import AppBackend


# ── Helpers ──────────────────────────────────────────────────────────────────

def _wav_dur(path: str) -> float:
    try:
        with _wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _quality(minutes: float) -> str:
    if minutes < 10:
        return "Poor"
    if minutes <= 20:
        return "Acceptable"
    return "Optimal"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Engine ────────────────────────────────────────────────────────────────────

class PrepareEngine:
    """Wraps AppBackend to expose the two-step pipeline."""

    def __init__(self, backend: AppBackend):
        self._b = backend

    # ------------------------------------------------------------------
    # Step 1 — blocking, safe to call from a sync FastAPI route
    # ------------------------------------------------------------------

    def prepare(
        self,
        files: list[str],
        model_name: str,
        max_workers: int = 4,
        resume: bool | None = None,
    ) -> dict:
        """
        Run preprocessing + LLM hyperparameter generation.
        Does NOT start training.
        Sets session status to 'awaiting_confirmation'.
        Returns {"ok": True, "session_id": ..., "hyperparams": {...}}.
        """
        log = get_logger(stage="prepare")

        # Auto-detect resume from existing checkpoints.
        if resume is None:
            resume = len(TrainingEngine().get_checkpoint_list(model_name)) > 0

        # Reuse an existing session dir when resuming.
        session_id: str | None = None
        if resume:
            for s in self._b.sessions.find_by_model(model_name):
                cid = s.get("session_id")
                if not cid:
                    continue
                if os.path.isdir(os.path.join(self._b.base_dir, "sessions_data", cid)):
                    session_id = cid
                    break

        if session_id is None:
            session_id = self._b.sessions.create_session(
                model_name=model_name,
                files=files,
                user_id=self._b.supabase.user_id,
            )

        session_base = os.path.join(self._b.base_dir, "sessions_data", session_id)
        preprocessor = AudioPreprocessor(base_dir=session_base)

        # ── Preprocess ────────────────────────────────────────────────
        self._b.sessions.set_status(session_id, "preprocessing")

        if resume:
            approved = preprocessor.get_approved_segment_paths()
            if approved:
                log.info("Skipping preprocessing — resume mode, approved segments present")
                total_s = sum(_wav_dur(p) for p in approved if os.path.exists(p))
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
        else:
            stats = preprocessor.process(files, max_workers=max_workers)
            preprocessor.approve_all()

        self._b.sessions.update_session(
            session_id,
            segment_manifest=preprocessor.manifest,
            status="preprocessing_done",
        )

        # ── Validate dataset ──────────────────────────────────────────
        dataset_path = os.path.join(session_base, "dataset", "segments")
        validation = self._b._validate_dataset(dataset_path)
        if not validation["ok"]:
            self._b.sessions.set_status(session_id, "error", validation["error"])
            return {"ok": False, "error": validation["error"], "session_id": session_id}

        # ── LLM hyperparameters ───────────────────────────────────────
        configurator = LLMConfigurator()
        hyperparams = configurator.get_hyperparameters(stats)

        self._b.sessions.update_session(
            session_id,
            hyperparams=hyperparams,
            total_epochs=hyperparams.get("epochs", 0),
            status="awaiting_confirmation",
        )

        log.success(f"Prepare complete — session={session_id}")
        return {"ok": True, "session_id": session_id, "hyperparams": hyperparams}

    # ------------------------------------------------------------------
    # Step 2 — async SSE generator
    # ------------------------------------------------------------------

    async def confirm_and_stream(
        self,
        session_id: str,
        hyperparams_override: dict | None = None,
    ):
        """
        Async generator yielding SSE strings.
        Merges stored hyperparams with any user-supplied overrides,
        then starts training via run_pipeline(resume=True, async_mode=True).

        Events emitted:
          training_started — {"session_id"}
          progress         — {"session_id", "status", "current_epoch", "total_epochs",
                              "latest_loss", "engine"}
          complete         — {"session_id", "status", "checkpoint_path"}
          error            — {"ok": false, "error": "..."}
        """
        session = self._b.sessions.get_session(session_id)
        if not session:
            yield _sse("error", {"ok": False, "error": f"Session {session_id} not found"})
            return

        allowed_statuses = {"awaiting_confirmation", "paused", "error"}
        if session.get("status") not in allowed_statuses:
            yield _sse("error", {
                "ok":    False,
                "error": (
                    f"Cannot confirm — session status is '{session['status']}'. "
                    f"Expected one of: {sorted(allowed_statuses)}"
                ),
            })
            return

        # User edits win over stored LLM values.
        stored = session.get("hyperparams") or {}
        final_hyperparams = {**stored, **(hyperparams_override or {})}

        loop = asyncio.get_event_loop()

        # Kick off training (resume=True skips re-preprocessing).
        result = await loop.run_in_executor(
            None,
            lambda: self._b.run_pipeline(
                files=session["files"],
                model_name=session["model_name"],
                hyperparams_override=final_hyperparams,
                async_mode=True,
                resume=True,
            ),
        )

        if not result.get("ok"):
            yield _sse("error", result)
            return

        yield _sse("training_started", {"session_id": session_id})

        terminal = {"done", "error"}
        last_epoch = -1

        while True:
            await asyncio.sleep(1)

            status_result = await loop.run_in_executor(
                None,
                lambda: self._b.get_session_status(session_id),
            )

            if not status_result.get("ok"):
                yield _sse("error", status_result)
                return

            sess   = status_result["session"]
            engine = status_result.get("engine", {})
            status = sess.get("status", "unknown")

            current_epoch = sess.get("current_epoch", 0)
            if current_epoch != last_epoch:
                last_epoch = current_epoch
                yield _sse(
                    "progress",
                    {
                        "session_id":    session_id,
                        "status":        status,
                        "current_epoch": current_epoch,
                        "total_epochs":  sess.get("total_epochs", 0),
                        "latest_loss":   sess.get("latest_loss"),
                        "engine":        engine,
                    },
                )

            if status in terminal:
                yield _sse(
                    "complete" if status == "done" else "error",
                    {
                        "session_id":      session_id,
                        "status":          status,
                        "checkpoint_path": sess.get("checkpoint_path"),
                        "error_message":   sess.get("error_message"),
                    },
                )
                return
