from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

FAKE_USER_ID = "uuid-user-auth-001"
FAKE_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.fake.signature"
FAKE_USER = {
    "id": FAKE_USER_ID,
    "email": "test@example.com",
    "role": "authenticated",
    "created_at": "2026-04-02T10:00:00+00:00",
}


def _make_supabase_mock(
    login_ok: bool = True,
    register_ok: bool = True,
    token: str = FAKE_TOKEN,
    user_id: str = FAKE_USER_ID,
    user_by_token: dict | None = None,
) -> MagicMock:
    sb = MagicMock()
    sb.login.return_value = {
        "ok": login_ok,
        "user_id": user_id if login_ok else None,
        "access_token": token if login_ok else None,
    }
    sb.register.return_value = {
        "ok": register_ok,
        "user_id": user_id if register_ok else None,
        "access_token": token if register_ok else None,
    }
    sb.logout.return_value = True
    sb.get_user_by_token.return_value = (
        user_by_token if user_by_token is not None else (FAKE_USER if login_ok else None)
    )
    return sb


def _make_backend_mock(supabase_mock: MagicMock) -> MagicMock:
    backend = MagicMock()
    backend.supabase = supabase_mock
    return backend


def _make_client(backend_mock: MagicMock) -> TestClient:
    with patch("api.main._backend", backend_mock), \
         patch("api.two_step_router._engine", MagicMock()):
        from api.app import app
        return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def auth_client():
    sb = _make_supabase_mock()
    backend = _make_backend_mock(sb)
    with patch("api.main._backend", backend), \
         patch("api.two_step_router._engine", MagicMock()):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, sb


class TestRegister:
    def test_register_success(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/register", json={
            "email": "new@example.com", "password": "Pass123!"
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["user_id"] == FAKE_USER_ID
        assert body["token"] == FAKE_TOKEN

    def test_register_failure(self, auth_client):
        client, sb = auth_client
        sb.register.return_value = {"ok": False}
        r = client.post("/auth/register", json={
            "email": "taken@example.com", "password": "Pass123!"
        })
        assert r.status_code == 400

    def test_register_missing_fields_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/register", json={"email": "a@b.com"})
        assert r.status_code == 422


class TestLogin:
    def test_login_success(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/login", json={
            "email": "test@example.com", "password": "correct"
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["token"] == FAKE_TOKEN
        assert body["user_id"] == FAKE_USER_ID

    def test_login_failure(self, auth_client):
        client, sb = auth_client
        sb.login.return_value = {"ok": False, "code": "invalid_credentials"}
        r = client.post("/auth/login", json={
            "email": "test@example.com", "password": "wrong"
        })
        assert r.status_code == 401

    def test_login_missing_fields_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/login", json={"email": "a@b.com"})
        assert r.status_code == 422


class TestMe:
    def _auth_header(self, token: str = FAKE_TOKEN) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def test_me_valid_token(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = FAKE_USER
        r = client.get("/auth/me", headers=self._auth_header())
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["user"]["id"] == FAKE_USER_ID

    def test_me_invalid_token(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = None
        r = client.get("/auth/me", headers=self._auth_header("invalid"))
        assert r.status_code == 401

    def test_me_missing_header(self, auth_client):
        client, _ = auth_client
        r = client.get("/auth/me")
        assert r.status_code == 401
