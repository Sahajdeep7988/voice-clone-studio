"""
Shared test fixtures and helpers.
Patching strategy:
  - api.main._backend  — the AppBackend singleton used by all main.py routes
  - api.two_step_router._engine — PrepareEngine singleton (also uses _backend)
Both are replaced with Mock objects before the FastAPI app processes any request.
"""

import json
import sys
import os
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TEST_INPUTS = "/home/sahajdeep-singh/voice-clone-studio/test_inputs"
TEST_MP3    = os.path.join(TEST_INPUTS, "test_audio_1.mp3")
TEST_WAV    = os.path.join(TEST_INPUTS, "test_audio_2.wav")
TEST_MKV    = os.path.join(TEST_INPUTS, "test_video_1.mkv")
TEST_MP4    = os.path.join(TEST_INPUTS, "test_video_1.mp4")
TEST_WEBM   = os.path.join(TEST_INPUTS, "test_video_2.webm")
TMP_MODEL  = "/tmp/test_model.pth"
if not os.path.exists(TMP_MODEL):
    try:
        with open(TMP_MODEL, "wb") as f:
            f.write(b"model")
    except OSError:
        pass

# ── Canonical mock data ──────────────────────────────────────────────────────

def make_session(
    session_id:   str  = "sess-abc-123",
    model_name:   str  = "test_model",
    status:       str  = "done",
    current_epoch: int = 50,
    total_epochs: int  = 100,
    latest_loss:  float = 0.0423,
    checkpoint:   str  = "/rvc/logs/test_model/test_model_50e_1000s.pth",
    hyperparams:  dict | None = None,
    user_id:      str  = "user-1",
) -> dict:
    return {
        "session_id":      session_id,
        "model_name":      model_name,
        "files":           [TEST_MP3],
        "hyperparams":     hyperparams or DEFAULT_HYPERPARAMS.copy(),
        "user_id":         user_id,
        "status":          status,
        "current_epoch":   current_epoch,
        "total_epochs":    total_epochs,
        "latest_loss":     latest_loss,
        "checkpoint_path": checkpoint,
        "segment_manifest": None,
        "error_message":   None,
        "created_at":      "2026-04-02T10:00:00+00:00",
        "updated_at":      "2026-04-02T10:05:00+00:00",
    }


DEFAULT_HYPERPARAMS = {
    "batch_size":          4,
    "epochs":              150,
    "learning_rate":       0.0001,
    "save_every_n_epochs": 10,
}

DEFAULT_ENGINE_STATUS = {
    "model_name":      "test_model",
    "is_running":      False,
    "checkpoints":     ["/rvc/logs/test_model/test_model_50e_1000s.pth"],
    "latest_epoch":    50,
    "extraction_done": True,
}


# ── Build a fully configured mock backend ────────────────────────────────────

def make_mock_backend(
    session: dict | None = None,
    sessions: list | None = None,
) -> MagicMock:
    s = session or make_session()
    sl = sessions or [s]

    mb = MagicMock()
    mb.supabase.get_user_by_token.return_value = {
        "id": "user-1",
        "email": "user-1@example.com",
        "role": "authenticated",
        "created_at": "2026-04-02T10:00:00+00:00",
    }
    mb.base_dir = ROOT

    # Sessions sub-object
    mb.sessions.get_session.return_value    = s
    mb.sessions.find_by_model.return_value  = []
    mb.sessions.create_session.return_value = s["session_id"]
    mb.sessions.list_sessions.return_value  = sl
    mb.sessions.set_status.return_value     = None
    mb.sessions.update_session.return_value = True
    mb.sessions.update_progress.return_value = None
    mb.sessions.set_checkpoint.return_value = None

    # Backend methods
    mb.list_sessions.return_value         = {"ok": True, "sessions": sl}
    mb.get_session_status.return_value    = {"ok": True, "session": s, "engine": DEFAULT_ENGINE_STATUS}
    mb.stop_training.return_value         = {"ok": True, "checkpoint": s["checkpoint_path"]}
    mb.convert_audio.return_value         = {"ok": True, "output_path": "/output/result.wav"}
    mb.run_pipeline.return_value          = {"ok": True, "session_id": s["session_id"], "async": True}
    mb._validate_dataset.return_value     = {"ok": True, "segment_count": 30}

    return mb


# ── SSE parsing helper ───────────────────────────────────────────────────────

def parse_sse(body: str) -> list[dict]:
    """Parse raw SSE body into list of {event, data} dicts."""
    events = []
    current = {}
    for line in body.splitlines():
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line.split(":", 1)[1].strip())
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ── A mock prepare engine that short-circuits all heavy compute ──────────────

def make_mock_prepare_engine(
    session: dict | None = None,
) -> MagicMock:
    s = session or make_session(status="awaiting_confirmation")
    mpe = MagicMock()
    mpe.prepare.return_value = {
        "ok":         True,
        "session_id": s["session_id"],
        "hyperparams": s["hyperparams"],
    }

    async def _fake_confirm_stream(session_id, hyperparams_override=None, **_):
        yield "event: training_started\ndata: " + json.dumps({"session_id": session_id}) + "\n\n"
        for epoch in (10, 25, 50):
            d = {"session_id": session_id, "status": "training",
                 "current_epoch": epoch, "total_epochs": 100, "latest_loss": 0.04}
            yield "event: progress\ndata: " + json.dumps(d) + "\n\n"
        d = {"session_id": session_id, "status": "done",
             "checkpoint_path": "/rvc/logs/test_model/model_100e.pth", "error_message": None}
        yield "event: complete\ndata: " + json.dumps(d) + "\n\n"

    mpe.confirm_and_stream.side_effect = _fake_confirm_stream
    return mpe


# ── Pytest fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_backend():
    return make_mock_backend()


@pytest.fixture
def mock_prepare_engine(mock_backend):
    return make_mock_prepare_engine()


@pytest.fixture
def client(mock_backend, mock_prepare_engine):
    """
    TestClient with both the backend singleton AND the prepare engine singleton
    patched out so no real I/O or compute happens.
    """
    with patch("api.main._backend", mock_backend), \
         patch("api.two_step_router._engine", mock_prepare_engine):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            c.headers.update({"Authorization": "Bearer test.token.value"})
            yield c
