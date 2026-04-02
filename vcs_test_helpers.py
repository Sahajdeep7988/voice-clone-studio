import json
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parent
TEST_INPUTS = ROOT / "test_inputs"


def make_session(
    session_id: str = "sess-abc-123",
    model_name: str = "test_model",
    status: str = "done",
    current_epoch: int = 50,
    total_epochs: int = 100,
    latest_loss: float = 0.0423,
    checkpoint: str = "/tmp/out/test_model_50e_1000s.pth",
    hyperparams: dict | None = None,
    user_id: str = "user-1",
) -> dict:
    return {
        "session_id": session_id,
        "model_name": model_name,
        "files": [str(TEST_INPUTS / "test_audio_1.mp3")],
        "hyperparams": hyperparams or {"epochs": 100},
        "user_id": user_id,
        "status": status,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "latest_loss": latest_loss,
        "checkpoint_path": checkpoint,
        "segment_manifest": None,
        "error_message": None,
        "created_at": "2026-04-02T10:00:00+00:00",
        "updated_at": "2026-04-02T10:05:00+00:00",
    }


def make_backend_mock(session: dict | None = None, sessions: list | None = None) -> MagicMock:
    s = session or make_session()
    sl = sessions or [s]

    mb = MagicMock()
    mb.base_dir = str(ROOT)

    mb.supabase.get_user_by_token.return_value = {
        "id": s["user_id"],
        "email": "user@example.com",
        "role": "authenticated",
        "created_at": "2026-04-02T10:00:00+00:00",
    }

    mb.sessions.get_session.return_value = s
    mb.sessions.create_session.return_value = s["session_id"]
    mb.sessions.list_sessions.return_value = sl
    mb.sessions.set_status.return_value = None
    mb.sessions.update_session.return_value = True
    mb.sessions.update_progress.return_value = None
    mb.sessions.set_checkpoint.return_value = None

    mb.list_sessions.return_value = {"ok": True, "sessions": sl}
    mb.get_session_status.return_value = {"ok": True, "session": s, "engine": {}}
    mb.stop_training.return_value = {"ok": True, "checkpoint": s["checkpoint_path"]}
    mb.convert_audio.return_value = {"ok": True, "output_path": "/tmp/out/result.wav"}
    mb.run_pipeline.return_value = {"ok": True, "session_id": s["session_id"], "async": True}
    mb._validate_dataset.return_value = {"ok": True, "segment_count": 30}

    return mb


def make_prepare_engine_mock(session: dict | None = None) -> MagicMock:
    s = session or make_session(status="awaiting_confirmation")
    mpe = MagicMock()
    mpe.prepare.return_value = {
        "ok": True,
        "session_id": s["session_id"],
        "hyperparams": s.get("hyperparams", {}),
    }

    async def _fake_confirm_stream(*_args, **_kwargs):
        yield f"event: training_started\ndata: {json.dumps({'session_id': s['session_id']})}\n\n"
        yield f"event: complete\ndata: {json.dumps({'session_id': s['session_id'], 'status': 'done'})}\n\n"

    mpe.confirm_and_stream.side_effect = _fake_confirm_stream
    return mpe


def parse_sse(body: str) -> list[dict]:
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
