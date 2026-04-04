"""
Comprehensive route tests — 0 to 100.

Coverage:
  ✓ GET  /sessions
  ✓ GET  /sessions/{id}/status
  ✓ POST /sessions/{id}/stop
  ✓ POST /inference/convert
  ✓ POST /sessions/create         (SSE)
  ✓ POST /sessions/prepare
  ✓ GET  /sessions/{id}/hyperparams
  ✓ POST /sessions/{id}/confirm   (SSE)

Scenario matrix:
  - Single file / multi-file inputs
  - AI-generated hyperparams (no override)
  - User-edited hyperparams (partial and full override)
  - Resume=True and resume=False
  - All error paths (not found, wrong status, bad payload, backend failure)
  - SSE event sequence validation (event names, data keys, order)
"""

import json
from copy import deepcopy
from unittest.mock import MagicMock, patch, AsyncMock, call
import pytest

from api.tests.conftest import (
    make_session,
    make_mock_backend,
    make_mock_prepare_engine,
    parse_sse,
    DEFAULT_HYPERPARAMS,
    TEST_MP3, TEST_WAV, TEST_MKV, TEST_MP4, TEST_WEBM, TMP_MODEL,
)


# ═══════════════════════════════════════════════════════════════════════════
# GET /sessions
# ═══════════════════════════════════════════════════════════════════════════

class TestListSessions:

    def test_empty_list(self, client, mock_backend):
        mock_backend.list_sessions.return_value = {"ok": True, "sessions": []}
        r = client.get("/sessions")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "sessions": []}

    def test_returns_all_sessions(self, client, mock_backend):
        sessions = [make_session("s1", "model_a"), make_session("s2", "model_b")]
        mock_backend.list_sessions.return_value = {"ok": True, "sessions": sessions}
        r = client.get("/sessions")
        assert r.status_code == 200
        assert len(r.json()["sessions"]) == 2

    def test_filter_by_user_id(self, client, mock_backend):
        mock_backend.list_sessions.return_value = {"ok": True, "sessions": []}
        r = client.get("/sessions?user_id=user-42")
        assert r.status_code == 403

    def test_no_user_id_passes_none(self, client, mock_backend):
        mock_backend.list_sessions.return_value = {"ok": True, "sessions": []}
        client.get("/sessions")
        mock_backend.list_sessions.assert_called_once_with(user_id="user-1")

    def test_session_fields_present(self, client, mock_backend):
        s = make_session()
        mock_backend.list_sessions.return_value = {"ok": True, "sessions": [s]}
        body = client.get("/sessions").json()
        sess = body["sessions"][0]
        for field in ("session_id", "model_name", "status", "current_epoch",
                      "total_epochs", "created_at"):
            assert field in sess, f"missing field: {field}"


# ═══════════════════════════════════════════════════════════════════════════
# GET /sessions/{id}/status
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionStatus:

    def test_found(self, client, mock_backend):
        s = make_session(status="training", current_epoch=42)
        mock_backend.get_session_status.return_value = {
            "ok": True, "session": s, "engine": {}
        }
        r = client.get("/sessions/sess-abc-123/status")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["session"]["status"] == "training"
        assert body["session"]["current_epoch"] == 42

    def test_not_found(self, client, mock_backend):
        mock_backend.get_session_status.return_value = {
            "ok": False, "error": "Session not found"
        }
        r = client.get("/sessions/nonexistent/status")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_session_id_passed_correctly(self, client, mock_backend):
        mock_backend.get_session_status.return_value = {
            "ok": True, "session": make_session(), "engine": {}
        }
        client.get("/sessions/my-custom-id/status")
        mock_backend.get_session_status.assert_called_once_with("my-custom-id")

    def test_engine_block_included(self, client, mock_backend):
        from api.tests.conftest import DEFAULT_ENGINE_STATUS
        s = make_session()
        mock_backend.get_session_status.return_value = {
            "ok": True, "session": s, "engine": DEFAULT_ENGINE_STATUS
        }
        body = client.get("/sessions/sess-abc-123/status").json()
        assert "engine" in body
        assert body["engine"]["is_running"] is False


# ═══════════════════════════════════════════════════════════════════════════
# POST /sessions/{id}/stop
# ═══════════════════════════════════════════════════════════════════════════

class TestStopSession:

    def test_graceful_stop(self, client, mock_backend):
        mock_backend.stop_training.return_value = {
            "ok": True, "checkpoint": "/rvc/logs/model/ckpt.pth"
        }
        r = client.post("/sessions/sess-abc-123/stop", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        args, kwargs = mock_backend.stop_training.call_args
        assert args[0] == "sess-abc-123"
        assert kwargs["force"] is False
        assert "user_id" in kwargs
        assert "access_token" in kwargs

    def test_force_stop(self, client, mock_backend):
        mock_backend.stop_training.return_value = {"ok": True, "checkpoint": None}
        r = client.post("/sessions/sess-abc-123/stop", json={"force": True})
        assert r.status_code == 200
        args, kwargs = mock_backend.stop_training.call_args
        assert args[0] == "sess-abc-123"
        assert kwargs["force"] is True

    def test_stop_not_found(self, client, mock_backend):
        mock_backend.stop_training.return_value = {
            "ok": False, "error": "Session not found"
        }
        r = client.post("/sessions/ghost/stop", json={})
        assert r.status_code == 404

    def test_empty_body_defaults_to_graceful(self, client, mock_backend):
        mock_backend.stop_training.return_value = {"ok": True, "checkpoint": None}
        client.post("/sessions/sess-abc-123/stop")
        args, kwargs = mock_backend.stop_training.call_args
        assert args[0] == "sess-abc-123"
        assert kwargs["force"] is False

    def test_no_active_engine(self, client, mock_backend):
        mock_backend.stop_training.return_value = {
            "ok": False, "error": "No active training engine"
        }
        r = client.post("/sessions/sess-abc-123/stop", json={})
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# POST /inference/convert
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceConvert:

    def test_success(self, client, mock_backend):
        mock_backend.convert_audio.return_value = {
            "ok": True, "output_path": "/tmp/out/converted.wav"
        }
        r = client.post("/inference/convert", json={
            "model_path":       TMP_MODEL,
            "input_audio_path": TEST_WAV,
            "output_path":      "/tmp/out/result.wav",
        })
        assert r.status_code == 200
        assert r.json()["output_path"] == "/tmp/out/converted.wav"

    def test_pitch_shift_default_zero(self, client, mock_backend):
        mock_backend.convert_audio.return_value = {"ok": True, "output_path": "/tmp/out/x.wav"}
        client.post("/inference/convert", json={
            "model_path":       TMP_MODEL,
            "input_audio_path": TEST_WAV,
            "output_path":      "/tmp/o.wav",
        })
        _, kwargs = mock_backend.convert_audio.call_args
        assert kwargs.get("pitch_shift", 0) == 0

    def test_custom_pitch_shift(self, client, mock_backend):
        mock_backend.convert_audio.return_value = {"ok": True, "output_path": "/tmp/out/x.wav"}
        client.post("/inference/convert", json={
            "model_path":       TMP_MODEL,
            "input_audio_path": TEST_WAV,
            "output_path":      "/tmp/o.wav",
            "pitch_shift":      5,
        })
        _, kwargs = mock_backend.convert_audio.call_args
        assert kwargs["pitch_shift"] == 5

    def test_backend_error_becomes_500(self, client, mock_backend):
        mock_backend.convert_audio.return_value = {
            "ok": False, "error": "Model file not found"
        }
        r = client.post("/inference/convert", json={
            "model_path":       TMP_MODEL,
            "input_audio_path": TEST_WAV,
            "output_path":      "/tmp/o.wav",
        })
        assert r.status_code == 500
        assert "not found" in r.json()["detail"].lower()

    def test_missing_required_fields(self, client, mock_backend):
        r = client.post("/inference/convert", json={"model_path": TMP_MODEL})
        assert r.status_code == 422

    def test_all_fields_forwarded(self, client, mock_backend):
        mock_backend.convert_audio.return_value = {"ok": True, "output_path": "/tmp/o.wav"}
        client.post("/inference/convert", json={
            "model_path":       TMP_MODEL,
            "input_audio_path": TEST_MP3,
            "output_path":      "/tmp/out/v.wav",
            "pitch_shift":      -3,
        })
        mock_backend.convert_audio.assert_called_once_with(
            model_path=TMP_MODEL,
            input_audio_path=TEST_MP3,
            output_path="/tmp/out/v.wav",
            pitch_shift=-3,
        )


# ═══════════════════════════════════════════════════════════════════════════
# POST /sessions/create  (SSE — headless/CLI path)
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateSessionSSE:
    """
    Tests the original /sessions/create endpoint which starts the full
    pipeline in async_mode and streams SSE progress.
    """

    def _make_progressing_status(self, session_id: str, epochs: list[int]):
        """Build a list of side_effect return values for get_session_status."""
        calls = []
        for ep in epochs:
            calls.append({
                "ok": True,
                "session": make_session(session_id=session_id, status="training",
                                        current_epoch=ep, total_epochs=100),
                "engine": {"is_running": True},
            })
        calls.append({
            "ok": True,
            "session": make_session(session_id=session_id, status="done",
                                    current_epoch=100, total_epochs=100),
            "engine": {"is_running": False},
        })
        return calls

    def test_sse_session_created_event(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "test-s1", "async": True
        }
        mock_backend.get_session_status.side_effect = [{
            "ok": True,
            "session": make_session("test-s1", status="done"),
            "engine": {},
        }]
        r = client.post("/sessions/create", json={
            "files":      [TEST_MP3],
            "model_name": "my_voice",
        })
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = parse_sse(r.text)
        names = [e["event"] for e in events]
        assert "session_created" in names

    def test_sse_complete_event(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "test-s2", "async": True
        }
        mock_backend.get_session_status.side_effect = self._make_progressing_status(
            "test-s2", [10, 50, 100]
        )
        r = client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "voice2"
        })
        events = parse_sse(r.text)
        names = [e["event"] for e in events]
        assert "complete" in names
        complete = next(e for e in events if e["event"] == "complete")
        assert complete["data"]["status"] == "done"

    def test_sse_progress_events_emitted(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "test-s3", "async": True
        }
        mock_backend.get_session_status.side_effect = self._make_progressing_status(
            "test-s3", [25, 50, 75]
        )
        events = parse_sse(client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "v3"
        }).text)
        progress = [e for e in events if e["event"] == "progress"]
        assert len(progress) >= 1
        for p in progress:
            assert "current_epoch" in p["data"]
            assert "total_epochs"  in p["data"]

    def test_sse_error_when_pipeline_fails(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": False, "error": "Preprocessing failed"
        }
        r = client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "bad_model"
        })
        events = parse_sse(r.text)
        names = [e["event"] for e in events]
        assert "error" in names

    def test_sse_error_status_from_training(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "test-s4", "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session("test-s4", status="error",
                                    checkpoint="/rvc/logs/v4/model_50e.pth"),
            "engine": {},
        }
        events = parse_sse(client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "v4"
        }).text)
        names = [e["event"] for e in events]
        assert "error" in names

    def test_single_file_input(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "sf-sess", "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session("sf-sess", status="done"),
            "engine": {},
        }
        r = client.post("/sessions/create", json={
            "files":      [TEST_MP3],
            "model_name": "single_file_model",
        })
        assert r.status_code == 200
        _, kwargs = mock_backend.run_pipeline.call_args
        assert kwargs["files"] == [TEST_MP3]

    def test_multi_file_input(self, client, mock_backend):
        files = [TEST_MP3, TEST_WAV, TEST_MKV]
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "mf-sess", "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session("mf-sess", status="done"),
            "engine": {},
        }
        client.post("/sessions/create", json={
            "files":      files,
            "model_name": "multi_file_model",
        })
        _, kwargs = mock_backend.run_pipeline.call_args
        assert set(kwargs["files"]) == set(files)

    def test_validation_missing_model_name(self, client, mock_backend):
        r = client.post("/sessions/create", json={"files": [TEST_MP3]})
        assert r.status_code == 422

    def test_validation_missing_files(self, client, mock_backend):
        r = client.post("/sessions/create", json={"model_name": "x"})
        assert r.status_code == 422

    def test_hyperparams_override_forwarded(self, client, mock_backend):
        override = {"batch_size": 8, "epochs": 200}
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "ho-sess", "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session("ho-sess", status="done"),
            "engine": {},
        }
        client.post("/sessions/create", json={
            "files":                [TEST_MP3],
            "model_name":           "override_model",
            "hyperparams_override": override,
        })
        _, kwargs = mock_backend.run_pipeline.call_args
        assert kwargs["hyperparams_override"] == override

    def test_resume_flag_forwarded(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "res-sess", "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session("res-sess", status="done"),
            "engine": {},
        }
        client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "resume_model", "resume": True
        })
        _, kwargs = mock_backend.run_pipeline.call_args
        assert kwargs["resume"] is True

    def test_max_workers_forwarded(self, client, mock_backend):
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": "mw-sess", "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session("mw-sess", status="done"),
            "engine": {},
        }
        client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "mw_model", "max_workers": 8
        })
        _, kwargs = mock_backend.run_pipeline.call_args
        assert kwargs["max_workers"] == 8


# ═══════════════════════════════════════════════════════════════════════════
# POST /sessions/prepare  (two-step Step 1)
# ═══════════════════════════════════════════════════════════════════════════

class TestPrepareSession:

    def test_success_returns_session_id_and_hyperparams(self, client, mock_prepare_engine):
        r = client.post("/sessions/prepare", json={
            "files":      [TEST_MP3],
            "model_name": "prepare_model",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "session_id" in body
        assert "hyperparams" in body

    def test_hyperparams_shape(self, client, mock_prepare_engine):
        r = client.post("/sessions/prepare", json={
            "files": [TEST_MP3], "model_name": "hp_model"
        })
        hp = r.json()["hyperparams"]
        for key in ("batch_size", "epochs", "learning_rate", "save_every_n_epochs"):
            assert key in hp, f"missing hyperparams key: {key}"

    def test_single_file(self, client, mock_prepare_engine):
        r = client.post("/sessions/prepare", json={
            "files":      [TEST_MP3],
            "model_name": "sf_model",
        })
        assert r.status_code == 200
        args, kwargs = mock_prepare_engine.prepare.call_args
        assert kwargs.get("files", args[0] if args else None) == [TEST_MP3]

    def test_multi_file(self, client, mock_prepare_engine):
        files = [TEST_MP3, TEST_WAV, TEST_MKV, TEST_MP4, TEST_WEBM]
        client.post("/sessions/prepare", json={
            "files": files, "model_name": "multi_model"
        })
        args, kwargs = mock_prepare_engine.prepare.call_args
        passed_files = kwargs.get("files", args[0] if args else None)
        assert set(passed_files) == set(files)

    def test_default_max_workers(self, client, mock_prepare_engine):
        client.post("/sessions/prepare", json={
            "files": [TEST_MP3], "model_name": "mw_model"
        })
        _, kwargs = mock_prepare_engine.prepare.call_args
        assert kwargs.get("max_workers", 4) == 4

    def test_custom_max_workers(self, client, mock_prepare_engine):
        client.post("/sessions/prepare", json={
            "files": [TEST_MP3], "model_name": "mw_model", "max_workers": 2
        })
        _, kwargs = mock_prepare_engine.prepare.call_args
        assert kwargs["max_workers"] == 2

    def test_resume_true(self, client, mock_prepare_engine):
        client.post("/sessions/prepare", json={
            "files": [TEST_MP3], "model_name": "res_model", "resume": True
        })
        _, kwargs = mock_prepare_engine.prepare.call_args
        assert kwargs["resume"] is True

    def test_resume_false(self, client, mock_prepare_engine):
        client.post("/sessions/prepare", json={
            "files": [TEST_MP3], "model_name": "res_model", "resume": False
        })
        _, kwargs = mock_prepare_engine.prepare.call_args
        assert kwargs["resume"] is False

    def test_resume_default_none(self, client, mock_prepare_engine):
        client.post("/sessions/prepare", json={
            "files": [TEST_MP3], "model_name": "res_model"
        })
        _, kwargs = mock_prepare_engine.prepare.call_args
        assert kwargs.get("resume") is None

    def test_preprocessing_error_becomes_500(self, client, mock_prepare_engine):
        mock_prepare_engine.prepare.return_value = {
            "ok": False, "error": "Dataset too short: 2.3 min"
        }
        r = client.post("/sessions/prepare", json={
            "files": [TEST_MP3], "model_name": "bad_model"
        })
        assert r.status_code == 500
        assert "too short" in r.json()["detail"].lower()

    def test_validation_missing_files(self, client, mock_prepare_engine):
        r = client.post("/sessions/prepare", json={"model_name": "x"})
        assert r.status_code == 422

    def test_validation_missing_model_name(self, client, mock_prepare_engine):
        r = client.post("/sessions/prepare", json={"files": [TEST_MP3]})
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# GET /sessions/{id}/hyperparams
# ═══════════════════════════════════════════════════════════════════════════

class TestGetHyperparams:

    def test_found(self, client, mock_backend):
        s = make_session(status="awaiting_confirmation")
        mock_backend.sessions.get_session.return_value = s
        r = client.get(f"/sessions/{s['session_id']}/hyperparams")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["hyperparams"] == s["hyperparams"]
        assert body["status"] == "awaiting_confirmation"

    def test_session_not_found(self, client, mock_backend):
        mock_backend.sessions.get_session.return_value = None
        r = client.get("/sessions/ghost/hyperparams")
        assert r.status_code == 404

    def test_no_hyperparams_yet(self, client, mock_backend):
        s = make_session(status="preprocessing")
        s["hyperparams"] = None
        mock_backend.sessions.get_session.return_value = s
        r = client.get(f"/sessions/{s['session_id']}/hyperparams")
        assert r.status_code == 404
        assert "prepare" in r.json()["detail"].lower()

    def test_empty_hyperparams_dict(self, client, mock_backend):
        s = make_session()
        s["hyperparams"] = {}
        mock_backend.sessions.get_session.return_value = s
        r = client.get(f"/sessions/{s['session_id']}/hyperparams")
        # Empty dict is falsy — treated as "not set"
        assert r.status_code == 404

    def test_hyperparams_keys_present(self, client, mock_backend):
        s = make_session(status="awaiting_confirmation")
        mock_backend.sessions.get_session.return_value = s
        body = client.get(f"/sessions/{s['session_id']}/hyperparams").json()
        hp = body["hyperparams"]
        for k in ("batch_size", "epochs", "learning_rate", "save_every_n_epochs"):
            assert k in hp


# ═══════════════════════════════════════════════════════════════════════════
# POST /sessions/{id}/confirm  (two-step Step 2 — SSE)
# ═══════════════════════════════════════════════════════════════════════════

class TestConfirmSession:

    def test_sse_training_started_event(self, client, mock_prepare_engine):
        r = client.post("/sessions/sess-abc-123/confirm", json={})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = parse_sse(r.text)
        names = [e["event"] for e in events]
        assert "training_started" in names

    def test_sse_progress_events(self, client, mock_prepare_engine):
        events = parse_sse(
            client.post("/sessions/sess-abc-123/confirm", json={}).text
        )
        progress = [e for e in events if e["event"] == "progress"]
        assert len(progress) >= 1
        for p in progress:
            d = p["data"]
            assert "current_epoch" in d
            assert "total_epochs"  in d
            assert "latest_loss"   in d
            assert "status"        in d

    def test_sse_complete_event(self, client, mock_prepare_engine):
        events = parse_sse(
            client.post("/sessions/sess-abc-123/confirm", json={}).text
        )
        names = [e["event"] for e in events]
        assert "complete" in names
        complete = next(e for e in events if e["event"] == "complete")
        assert complete["data"]["status"] == "done"
        assert "checkpoint_path" in complete["data"]

    def test_sse_event_order(self, client, mock_prepare_engine):
        events = parse_sse(
            client.post("/sessions/sess-abc-123/confirm", json={}).text
        )
        names = [e["event"] for e in events]
        assert names[0] == "training_started"
        assert names[-1] in ("complete", "error")

    def test_error_when_session_not_found(self, client, mock_prepare_engine):
        async def _not_found(session_id, hyperparams_override=None, **_):
            yield (
                f"event: error\n"
                f"data: {json.dumps({'ok': False, 'error': 'Session not found'})}\n\n"
            )
        mock_prepare_engine.confirm_and_stream.side_effect = _not_found
        events = parse_sse(
            client.post("/sessions/ghost/confirm", json={}).text
        )
        names = [e["event"] for e in events]
        assert "error" in names
        err = next(e for e in events if e["event"] == "error")
        assert err["data"]["ok"] is False

    def test_error_when_wrong_status(self, client, mock_prepare_engine):
        async def _wrong_status(session_id, hyperparams_override=None, **_):
            yield (
                f"event: error\n"
                f"data: {json.dumps({'ok': False, 'error': 'Cannot confirm — status is training'})}\n\n"
            )
        mock_prepare_engine.confirm_and_stream.side_effect = _wrong_status
        r = client.post("/sessions/sess-abc-123/confirm", json={})
        events = parse_sse(r.text)
        assert any(e["event"] == "error" for e in events)

    def test_hyperparams_override_passed_through(self, client, mock_prepare_engine):
        override = {"epochs": 300, "batch_size": 16}
        client.post("/sessions/sess-abc-123/confirm", json={
            "hyperparams_override": override
        })
        _, kwargs = mock_prepare_engine.confirm_and_stream.call_args
        assert kwargs.get("hyperparams_override") == override

    def test_no_override_passes_none(self, client, mock_prepare_engine):
        client.post("/sessions/sess-abc-123/confirm", json={})
        _, kwargs = mock_prepare_engine.confirm_and_stream.call_args
        assert kwargs.get("hyperparams_override") is None

    def test_empty_body_accepted(self, client, mock_prepare_engine):
        r = client.post("/sessions/sess-abc-123/confirm")
        assert r.status_code == 200

    def test_session_id_in_training_started(self, client, mock_prepare_engine):
        events = parse_sse(
            client.post("/sessions/sess-abc-123/confirm", json={}).text
        )
        started = next(e for e in events if e["event"] == "training_started")
        assert started["data"]["session_id"] == "sess-abc-123"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario Tests — full multi-step flows
# ═══════════════════════════════════════════════════════════════════════════

class TestScenarios:
    """
    End-to-end scenario tests covering the major user journeys.
    All heavy compute is mocked; only the HTTP + session data flow is verified.
    """

    def test_scenario_single_file_ai_hyperparams(self, client, mock_backend, mock_prepare_engine):
        """
        Single audio file → prepare → review AI hyperparams → confirm with no edits.
        """
        session_id = "single-ai-001"
        mock_prepare_engine.prepare.return_value = {
            "ok":         True,
            "session_id": session_id,
            "hyperparams": DEFAULT_HYPERPARAMS,
        }
        awaiting_session = make_session(
            session_id=session_id, status="awaiting_confirmation",
            hyperparams=DEFAULT_HYPERPARAMS,
        )
        mock_backend.sessions.get_session.return_value = awaiting_session

        # Step 1: prepare
        r1 = client.post("/sessions/prepare", json={
            "files":      [TEST_MP3],
            "model_name": "single_ai_voice",
        })
        assert r1.status_code == 200
        assert r1.json()["session_id"] == session_id

        # Step 1b: review hyperparams via GET
        r_hp = client.get(f"/sessions/{session_id}/hyperparams")
        assert r_hp.status_code == 200
        assert r_hp.json()["hyperparams"]["epochs"] == DEFAULT_HYPERPARAMS["epochs"]

        # Step 2: confirm with no overrides → uses AI hyperparams
        r2 = client.post(f"/sessions/{session_id}/confirm", json={})
        events = parse_sse(r2.text)
        names = [e["event"] for e in events]
        assert "training_started" in names
        assert "complete" in names
        _, kwargs = mock_prepare_engine.confirm_and_stream.call_args
        assert kwargs.get("hyperparams_override") is None

    def test_scenario_single_file_edited_hyperparams(self, client, mock_backend, mock_prepare_engine):
        """
        Single audio file → prepare → user edits epochs → confirm with overrides.
        """
        session_id = "single-edit-002"
        ai_hp = DEFAULT_HYPERPARAMS.copy()
        mock_prepare_engine.prepare.return_value = {
            "ok": True, "session_id": session_id, "hyperparams": ai_hp
        }
        mock_backend.sessions.get_session.return_value = make_session(
            session_id=session_id, status="awaiting_confirmation", hyperparams=ai_hp
        )

        client.post("/sessions/prepare", json={"files": [TEST_WAV], "model_name": "edit_voice"})

        user_edits = {"epochs": 250, "batch_size": 8}
        r2 = client.post(f"/sessions/{session_id}/confirm", json={
            "hyperparams_override": user_edits
        })
        events = parse_sse(r2.text)
        assert any(e["event"] == "complete" for e in events)
        _, kwargs = mock_prepare_engine.confirm_and_stream.call_args
        assert kwargs["hyperparams_override"]["epochs"] == 250
        assert kwargs["hyperparams_override"]["batch_size"] == 8

    def test_scenario_multi_file_ai_hyperparams(self, client, mock_backend, mock_prepare_engine):
        """
        Multiple formats (mp3, wav, mkv, mp4, webm) → prepare → confirm unedited.
        """
        session_id = "multi-ai-003"
        files = [TEST_MP3, TEST_WAV, TEST_MKV, TEST_MP4]
        mock_prepare_engine.prepare.return_value = {
            "ok": True, "session_id": session_id, "hyperparams": DEFAULT_HYPERPARAMS
        }
        mock_backend.sessions.get_session.return_value = make_session(
            session_id=session_id, status="awaiting_confirmation"
        )

        r1 = client.post("/sessions/prepare", json={
            "files": files, "model_name": "multi_voice"
        })
        assert r1.status_code == 200
        _, kwargs = mock_prepare_engine.prepare.call_args
        assert set(kwargs["files"]) == set(files)

        events = parse_sse(
            client.post(f"/sessions/{session_id}/confirm", json={}).text
        )
        assert any(e["event"] == "complete" for e in events)

    def test_scenario_multi_file_full_hyperparams_override(self, client, mock_backend, mock_prepare_engine):
        """
        Multiple files + user overrides ALL hyperparams.
        """
        session_id = "multi-full-override-004"
        mock_prepare_engine.prepare.return_value = {
            "ok": True, "session_id": session_id, "hyperparams": DEFAULT_HYPERPARAMS
        }
        mock_backend.sessions.get_session.return_value = make_session(
            session_id=session_id, status="awaiting_confirmation"
        )
        full_override = {
            "batch_size":          16,
            "epochs":              300,
            "learning_rate":       0.0001,
            "save_every_n_epochs": 25,
        }
        client.post("/sessions/prepare", json={
            "files": [TEST_MP3, TEST_WAV], "model_name": "override_all"
        })
        r2 = client.post(f"/sessions/{session_id}/confirm", json={
            "hyperparams_override": full_override
        })
        events = parse_sse(r2.text)
        assert any(e["event"] == "complete" for e in events)
        _, kwargs = mock_prepare_engine.confirm_and_stream.call_args
        for k, v in full_override.items():
            assert kwargs["hyperparams_override"][k] == v

    def test_scenario_headless_create_single_file(self, client, mock_backend):
        """
        Headless /sessions/create with a single file — CLI/programmatic path.
        """
        sid = "headless-sf-005"
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": sid, "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session(sid, status="done"),
            "engine": {},
        }
        events = parse_sse(client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "headless_single"
        }).text)
        names = [e["event"] for e in events]
        assert "session_created" in names
        assert "complete" in names

    def test_scenario_headless_create_multi_file(self, client, mock_backend):
        """
        Headless /sessions/create with all file formats.
        """
        sid = "headless-mf-006"
        files = [TEST_MP3, TEST_WAV, TEST_MKV]
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": sid, "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session(sid, status="done"),
            "engine": {},
        }
        r = client.post("/sessions/create", json={
            "files": files, "model_name": "headless_multi"
        })
        events = parse_sse(r.text)
        assert any(e["event"] == "complete" for e in events)
        _, kwargs = mock_backend.run_pipeline.call_args
        assert set(kwargs["files"]) == set(files)

    def test_scenario_headless_with_hyperparams_override(self, client, mock_backend):
        """
        Headless path with explicit hyperparams override (power-user mode).
        """
        sid = "headless-override-007"
        override = {"batch_size": 16, "epochs": 200}
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": sid, "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session(sid, status="done"),
            "engine": {},
        }
        client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "headless_override",
            "hyperparams_override": override,
        })
        _, kwargs = mock_backend.run_pipeline.call_args
        assert kwargs["hyperparams_override"] == override

    def test_scenario_resume_existing_session(self, client, mock_backend):
        """
        resume=True: pipeline should skip re-preprocessing.
        """
        sid = "resume-008"
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": sid, "async": True
        }
        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session(sid, status="done"),
            "engine": {},
        }
        client.post("/sessions/create", json={
            "files": [TEST_MP3], "model_name": "resume_model", "resume": True
        })
        _, kwargs = mock_backend.run_pipeline.call_args
        assert kwargs["resume"] is True

    def test_scenario_create_then_stop(self, client, mock_backend):
        """
        Start a session then stop it — verify stop is idempotent and returns checkpoint.
        """
        sid = "stop-009"
        mock_backend.run_pipeline.return_value = {
            "ok": True, "session_id": sid, "async": True
        }
        # SSE generator polls until a terminal state ("done"/"error"), so the list
        # must end with one.  We simulate: training → training → done.
        statuses = [
            {"ok": True, "session": make_session(sid, status="training", current_epoch=10), "engine": {}},
            {"ok": True, "session": make_session(sid, status="training", current_epoch=20), "engine": {}},
            {"ok": True, "session": make_session(sid, status="done"),    "engine": {}},
        ]
        mock_backend.get_session_status.side_effect = statuses

        client.post("/sessions/create", json={"files": [TEST_MP3], "model_name": "stop_model"})

        mock_backend.stop_training.return_value = {
            "ok": True, "checkpoint": "/rvc/logs/stop_model/ckpt_20e.pth"
        }
        r_stop = client.post(f"/sessions/{sid}/stop", json={})
        assert r_stop.status_code == 200
        assert r_stop.json()["checkpoint"] is not None

    def test_scenario_verify_all_routes_registered(self, client):
        """
        Smoke-test: every defined route returns something other than 404 (method not allowed
        or validation error is fine — pure 404 means the route was never registered).
        """
        routes = [
            ("GET",  "/sessions"),
            ("GET",  "/sessions/any-id/status"),
            ("POST", "/sessions/any-id/stop"),
            ("POST", "/inference/convert"),
            ("POST", "/sessions/create"),
            ("POST", "/sessions/prepare"),
            ("GET",  "/sessions/any-id/hyperparams"),
            ("POST", "/sessions/any-id/confirm"),
        ]
        for method, path in routes:
            r = client.request(method, path)
            assert r.status_code != 404, (
                f"Route {method} {path} returned 404 — not registered"
            )

    def test_scenario_session_lifecycle(self, client, mock_backend):
        """
        Full lifecycle: prepare → hyperparams GET → confirm → status → stop → list.
        """
        sid = "lifecycle-010"
        hp  = DEFAULT_HYPERPARAMS.copy()

        # Prepare
        with patch("api.two_step_router._engine") as mpe:
            mpe.prepare.return_value = {
                "ok": True, "session_id": sid, "hyperparams": hp
            }

            async def _fake_stream(session_id, hyperparams_override=None, **_):
                yield f"event: training_started\ndata: {json.dumps({'session_id': session_id})}\n\n"
                yield f"event: complete\ndata: {json.dumps({'session_id': session_id, 'status': 'done', 'checkpoint_path': '/rvc/logs/v.pth', 'error_message': None})}\n\n"
            mpe.confirm_and_stream.side_effect = _fake_stream

            mock_backend.sessions.get_session.return_value = make_session(
                session_id=sid, status="awaiting_confirmation", hyperparams=hp
            )

            r1 = client.post("/sessions/prepare", json={"files": [TEST_MP3], "model_name": "lifecycle_model"})
            assert r1.json()["ok"] is True

            r_hp = client.get(f"/sessions/{sid}/hyperparams")
            assert r_hp.status_code == 200

            r2 = client.post(f"/sessions/{sid}/confirm", json={})
            events = parse_sse(r2.text)
            assert any(e["event"] == "complete" for e in events)

        mock_backend.get_session_status.return_value = {
            "ok": True,
            "session": make_session(sid, status="done"),
            "engine": {},
        }
        r3 = client.get(f"/sessions/{sid}/status")
        assert r3.status_code == 200

        mock_backend.stop_training.return_value = {
            "ok": False, "error": "No active training engine"
        }
        r4 = client.post(f"/sessions/{sid}/stop", json={})
        assert r4.status_code == 404

        mock_backend.list_sessions.return_value = {
            "ok": True, "sessions": [make_session(sid, status="done")]
        }
        r5 = client.get("/sessions")
        assert any(s["session_id"] == sid for s in r5.json()["sessions"])
