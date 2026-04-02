from unittest.mock import MagicMock

import pytest

from vcs_test_helpers import parse_sse


def test_sse_stream_progress_done(client, backend_mock, fast_no_sleep, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    backend_mock.run_pipeline.return_value = {"ok": True, "session_id": "sess-1", "async": True}
    backend_mock.get_session_status.side_effect = [
        {"ok": True, "session": {"session_id": "sess-1", "status": "training", "current_epoch": 1, "total_epochs": 3, "latest_loss": 0.1}, "engine": {}},
        {"ok": True, "session": {"session_id": "sess-1", "status": "done", "current_epoch": 3, "total_epochs": 3, "latest_loss": 0.05, "checkpoint_path": "/tmp/out.pth"}, "engine": {}},
    ]

    r = client.post("/sessions/create", json={
        "files": [str(audio)],
        "model_name": "m1",
    })
    events = parse_sse(r.text)
    assert events[0]["event"] == "session_created"
    assert any(e["event"] == "progress" for e in events)
    assert events[-1]["event"] == "complete"


def test_sse_stream_error(client, backend_mock, fast_no_sleep, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    backend_mock.run_pipeline.return_value = {"ok": True, "session_id": "sess-1", "async": True}
    backend_mock.get_session_status.side_effect = [
        {"ok": True, "session": {"session_id": "sess-1", "status": "error", "current_epoch": 0, "total_epochs": 3, "latest_loss": None, "error_message": "fail"}, "engine": {}},
    ]
    r = client.post("/sessions/create", json={
        "files": [str(audio)],
        "model_name": "m1",
    })
    events = parse_sse(r.text)
    assert events[-1]["event"] == "error"


def test_sse_stream_paused(client, backend_mock, fast_no_sleep, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    backend_mock.run_pipeline.return_value = {"ok": True, "session_id": "sess-1", "async": True}
    backend_mock.get_session_status.side_effect = [
        {"ok": True, "session": {"session_id": "sess-1", "status": "paused", "current_epoch": 2, "total_epochs": 3, "latest_loss": 0.2}, "engine": {}},
    ]
    r = client.post("/sessions/create", json={
        "files": [str(audio)],
        "model_name": "m1",
    })
    events = parse_sse(r.text)
    assert events[-1]["event"] == "paused"


def test_confirm_sse_passthrough(client, prepare_engine_mock):
    r = client.post("/sessions/sess-abc-123/confirm", json={"hyperparams_override": {}})
    events = parse_sse(r.text)
    assert events[0]["event"] == "training_started"
    assert events[-1]["event"] in ("complete", "error")
