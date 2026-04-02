"""
Session Manager
Persists pipeline session state to disk so runs survive restarts.
Each session = one model training job.
State stored at ~/.voice_clone_studio/sessions/{session_id}.json
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


SESSIONS_DIR = os.path.expanduser("~/.voice_clone_studio/sessions")


class SessionManager:

    def __init__(self, sessions_dir: str = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        os.makedirs(self.sessions_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        model_name: str,
        files: list,
        hyperparams: dict | None = None,
        user_id: str | None = None,
    ) -> str:
        """Create a new session and return its session_id."""
        session_id = str(uuid.uuid4())
        state = {
            "session_id":      session_id,
            "model_name":      model_name,
            "files":           files,
            "hyperparams":     hyperparams or {},
            "user_id":         user_id,
            "status":          "created",     # created|preprocessing|training|paused|done|error
            "current_epoch":   0,
            "total_epochs":    hyperparams.get("epochs", 0) if hyperparams else 0,
            "latest_loss":     None,
            "checkpoint_path": None,
            "segment_manifest": None,
            "error_message":   None,
            "created_at":      _now(),
            "updated_at":      _now(),
        }
        self._write(session_id, state)
        print(f"[Session] Created session {session_id} for model '{model_name}'")
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def update_session(self, session_id: str, **kwargs) -> bool:
        state = self.get_session(session_id)
        if state is None:
            return False
        state.update(kwargs)
        state["updated_at"] = _now()
        self._write(session_id, state)
        return True

    def delete_session(self, session_id: str) -> bool:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_sessions(self, user_id: str | None = None) -> list:
        """Return all sessions, optionally filtered by user_id, newest first."""
        sessions = []
        for p in Path(self.sessions_dir).glob("*.json"):
            try:
                with open(p) as f:
                    s = json.load(f)
                if user_id is None or s.get("user_id") == user_id:
                    sessions.append(s)
            except Exception:
                pass
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    def get_active_sessions(self) -> list:
        return [s for s in self.list_sessions()
                if s["status"] in ("preprocessing", "training")]

    def find_by_model(self, model_name: str) -> list:
        return [s for s in self.list_sessions()
                if s.get("model_name") == model_name]

    # ------------------------------------------------------------------
    # Convenience status transitions
    # ------------------------------------------------------------------

    def set_status(self, session_id: str, status: str,
                   error: str | None = None) -> None:
        kwargs: dict = {"status": status}
        if error:
            kwargs["error_message"] = error
        self.update_session(session_id, **kwargs)

    def update_progress(self, session_id: str, epoch: int, loss: float) -> None:
        self.update_session(session_id,
                            current_epoch=epoch,
                            latest_loss=loss,
                            status="training")

    def set_checkpoint(self, session_id: str, checkpoint_path: str) -> None:
        self.update_session(session_id, checkpoint_path=checkpoint_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, f"{session_id}.json")

    def _write(self, session_id: str, state: dict) -> None:
        with open(self._path(session_id), "w") as f:
            json.dump(state, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
