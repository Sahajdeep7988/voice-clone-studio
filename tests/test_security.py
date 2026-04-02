from unittest.mock import MagicMock, patch

import pytest

from vcs_test_helpers import make_backend_mock, make_prepare_engine_mock, make_session


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("get", "/sessions", None),
        ("get", "/sessions/sess-abc-123/status", None),
        ("post", "/sessions/sess-abc-123/stop", {"force": False}),
        ("post", "/sessions/create", {"files": ["/tmp/a.wav"], "model_name": "m"}),
        ("post", "/sessions/prepare", {"files": ["/tmp/a.wav"], "model_name": "m"}),
        ("post", "/sessions/sess-abc-123/confirm", {"hyperparams_override": {}}),
        ("get", "/sessions/sess-abc-123/hyperparams", None),
        ("post", "/inference/convert", {
            "model_path": "/tmp/m.pth", "input_audio_path": "/tmp/in.wav", "output_path": "/tmp/out.wav"
        }),
    ],
)
def test_auth_protection_all_routes(unauth_client, tmp_path, method, path, payload):
    for p in ("a.wav", "in.wav", "m.pth"):
        (tmp_path / p).write_bytes(b"abc")
    path = path.replace("/tmp/a.wav", str(tmp_path / "a.wav"))
    path = path.replace("/tmp/in.wav", str(tmp_path / "in.wav"))
    path = path.replace("/tmp/m.pth", str(tmp_path / "m.pth"))
    if payload:
        payload = {k: (str(tmp_path / "a.wav") if v == ["/tmp/a.wav"] else v) for k, v in payload.items()}
        if "input_audio_path" in payload:
            payload["input_audio_path"] = str(tmp_path / "in.wav")
        if "model_path" in payload:
            payload["model_path"] = str(tmp_path / "m.pth")
        if "output_path" in payload:
            payload["output_path"] = str(tmp_path / "out.wav")

    r = getattr(unauth_client, method)(path, json=payload) if payload else getattr(unauth_client, method)(path)
    assert r.status_code == 401


def test_multi_user_isolation_403(tmp_path):
    session = make_session(user_id="user-a")
    backend = make_backend_mock(session=session)
    prepare_engine = make_prepare_engine_mock(session=session)

    def _user_for_token(token: str):
        if token == "token-a":
            return {"id": "user-a", "email": "a@example.com", "role": "authenticated", "created_at": "2026-04-02T10:00:00+00:00"}
        if token == "token-b":
            return {"id": "user-b", "email": "b@example.com", "role": "authenticated", "created_at": "2026-04-02T10:00:00+00:00"}
        return None

    backend.supabase.get_user_by_token.side_effect = _user_for_token

    with patch("api.main._backend", backend), patch("api.two_step_router._engine", prepare_engine):
        from api.app import app
        from starlette.testclient import TestClient

        with TestClient(app, raise_server_exceptions=True) as c:
            c.headers.update({"Authorization": "Bearer token-b"})
            r = c.get("/sessions/sess-abc-123/status")
            assert r.status_code == 403

            r = c.post("/sessions/sess-abc-123/stop", json={"force": False})
            assert r.status_code == 403

            r = c.get("/sessions", params={"user_id": "user-a"})
            assert r.status_code == 403
