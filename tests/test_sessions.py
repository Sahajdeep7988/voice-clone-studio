from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from vcs_test_helpers import parse_sse


def test_create_session_valid(client, tmp_path, fast_no_sleep):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    r = client.post("/sessions/create", json={
        "files": [str(audio)],
        "model_name": "m1",
    })
    assert r.status_code == 200


def test_create_session_invalid_input_422(client):
    r = client.post("/sessions/create", json={"model_name": "m1"})
    assert r.status_code == 422


def test_invalid_json_400(client):
    r = client.post(
        "/sessions/create",
        content="{bad json",
        headers={"Content-Type": "application/json", "Authorization": "Bearer test.token.value"},
    )
    assert r.status_code in (400, 422)


def test_get_session_status(client):
    r = client.get("/sessions/sess-abc-123/status")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_stop_session(client):
    r = client.post("/sessions/sess-abc-123/stop", json={"force": False})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_resume_via_confirm(client):
    r = client.post("/sessions/sess-abc-123/confirm", json={"hyperparams_override": {"epochs": 10}})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[0]["event"] == "training_started"


def test_file_safety_rejects_invalid_paths(client):
    r = client.post("/inference/convert", json={
        "model_path": "/etc/passwd",
        "input_audio_path": "/etc/hosts",
        "output_path": "/tmp/out.wav",
    })
    assert r.status_code in (400, 403)


def test_file_safety_allows_tmp_paths(client, tmp_path):
    model = tmp_path / "m.pth"
    audio = tmp_path / "in.wav"
    model.write_bytes(b"model")
    audio.write_bytes(b"audio")
    r = client.post("/inference/convert", json={
        "model_path": str(model),
        "input_audio_path": str(audio),
        "output_path": str(tmp_path / "out.wav"),
    })
    assert r.status_code == 200


def test_duplicate_sessions_allowed(client, backend_mock, tmp_path, fast_no_sleep):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")
    backend_mock.run_pipeline.reset_mock()
    r1 = client.post("/sessions/create", json={"files": [str(audio)], "model_name": "m1"})
    r2 = client.post("/sessions/create", json={"files": [str(audio)], "model_name": "m1"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert backend_mock.run_pipeline.call_count == 2


def test_concurrent_requests_basic_simulation(backend_mock, prepare_engine_mock, tmp_path, fast_no_sleep):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"abc")

    def _call():
        with patch("api.main._backend", backend_mock), patch("api.two_step_router._engine", prepare_engine_mock):
            from api.app import app
            from starlette.testclient import TestClient
            with TestClient(app, raise_server_exceptions=True) as c:
                c.headers.update({"Authorization": "Bearer test.token.value"})
                return c.post("/sessions/create", json={
                    "files": [str(audio)],
                    "model_name": "m1",
                }).status_code

    backend_mock.run_pipeline.reset_mock()
    with ThreadPoolExecutor(max_workers=2) as ex:
        codes = list(ex.map(lambda _: _call(), range(2)))
    assert codes == [200, 200]
    assert backend_mock.run_pipeline.call_count == 2


def test_supabase_sync_payload(monkeypatch, tmp_path):
    from services.backend import AppBackend

    class FakeSessions:
        def __init__(self):
            self.session = {
                "session_id": "sess-1",
                "model_name": "m1",
                "files": [str(tmp_path / "a.wav")],
                "hyperparams": {},
                "user_id": "user-1",
                "status": "created",
                "current_epoch": 0,
                "total_epochs": 0,
                "latest_loss": None,
                "checkpoint_path": None,
                "segment_manifest": None,
                "error_message": None,
            }

        def create_session(self, model_name, files, user_id=None):
            self.session["model_name"] = model_name
            self.session["files"] = files
            self.session["user_id"] = user_id
            return self.session["session_id"]

        def get_session(self, session_id):
            return self.session if session_id == self.session["session_id"] else None

        def update_session(self, session_id, **kwargs):
            self.session.update(kwargs)
            return True

        def set_status(self, session_id, status, error=None):
            self.session["status"] = status
            if error:
                self.session["error_message"] = error

        def update_progress(self, session_id, epoch, loss):
            self.session["current_epoch"] = epoch
            self.session["latest_loss"] = loss

        def set_checkpoint(self, session_id, checkpoint_path):
            self.session["checkpoint_path"] = checkpoint_path

    class FakePreprocessor:
        def __init__(self, base_dir):
            self.manifest = str(tmp_path / "manifest.json")

        def process(self, files, max_workers=4):
            return {
                "segment_paths": files,
                "total_duration_minutes": 1.0,
                "segment_count": len(files),
                "quality_label": "Poor",
                "manifest_path": self.manifest,
                "total_files": len(files),
                "success_count": len(files),
                "failed_count": 0,
                "failures": [],
            }

        def approve_all(self):
            return True

        def get_approved_segment_paths(self):
            return []

    class FakeConfigurator:
        def get_hyperparameters(self, _stats):
            return {"epochs": 1}

        def _scan_hardware(self):
            return {"gpu": "test"}

    class FakeTrainingEngine:
        def get_checkpoint_list(self, _model_name):
            return []

        def start_training(self, **_kwargs):
            return str(tmp_path / "out.pth")

    backend = AppBackend(base_dir=str(tmp_path))
    backend.sessions = FakeSessions()
    backend.supabase = MagicMock()
    backend.supabase.sync_training_session.return_value = True

    monkeypatch.setattr("services.backend.AudioPreprocessor", FakePreprocessor)
    monkeypatch.setattr("services.backend.LLMConfigurator", FakeConfigurator)
    monkeypatch.setattr("services.backend.TrainingEngine", FakeTrainingEngine)
    monkeypatch.setattr(backend, "_validate_dataset", lambda _p: {"ok": True, "segment_count": 1})

    (tmp_path / "a.wav").write_bytes(b"abc")
    result = backend.run_pipeline(
        files=[str(tmp_path / "a.wav")],
        model_name="m1",
        async_mode=False,
        user_id="user-1",
        access_token="token-1",
    )
    assert result["ok"] is True
    assert backend.supabase.sync_training_session.called
    args, kwargs = backend.supabase.sync_training_session.call_args
    payload = args[0]
    assert payload["user_id"] == "user-1"
    assert payload["session_id"] == "sess-1"
    assert payload["model_name"] == "m1"
    assert kwargs["access_token"] == "token-1"
