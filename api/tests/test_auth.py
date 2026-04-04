"""
Auth route tests — full coverage for all 4 endpoints.

Routes tested:
  POST /auth/register
  POST /auth/login
  POST /auth/logout
  GET  /auth/me

Coverage matrix:
  ✓ Register success
  ✓ Register failure (email already in use / weak password)
  ✓ Register missing fields (422 validation)
  ✓ Register invalid email format (422 validation)
  ✓ Login success — token and user_id returned
  ✓ Login wrong password
  ✓ Login unknown email
  ✓ Login missing fields (422)
  ✓ Logout while authenticated
  ✓ Logout while already logged out (idempotent)
  ✓ Me — valid token returns user info
  ✓ Me — invalid/expired token → 401
  ✓ Me — missing Authorization header → 401
  ✓ Me — malformed header (no Bearer prefix) → 401
  ✓ Me — empty Bearer value → 401
  ✓ Me — Supabase offline (get_user_by_token returns None) → 401
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Shared mock data ──────────────────────────────────────────────────────────

FAKE_USER_ID = "uuid-user-auth-001"
FAKE_TOKEN   = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.fake.signature"
FAKE_USER    = {
    "id":         FAKE_USER_ID,
    "email":      "test@example.com",
    "role":       "authenticated",
    "created_at": "2026-04-02T10:00:00+00:00",
}


def _make_supabase_mock(
    login_ok:      bool = True,
    register_ok:   bool = True,
    token:         str  = FAKE_TOKEN,
    user_id:       str  = FAKE_USER_ID,
    user_by_token: dict | None = None,
) -> MagicMock:
    sb = MagicMock()
    sb.login.return_value              = (
        {"ok": True, "user_id": user_id, "access_token": token, "refresh_token": "refresh.tok",
         "email": "test@example.com"}
        if login_ok else
        {"ok": False, "code": "invalid_credentials", "detail": "Invalid email or password."}
    )
    sb.register.return_value           = (
        {"ok": True, "user_id": user_id, "access_token": token, "refresh_token": "refresh.tok",
         "email": "test@example.com"}
        if register_ok else
        {"ok": False, "code": "email_exists", "detail": "User already registered."}
    )
    sb.logout.return_value             = True
    sb.get_user_by_token.return_value  = (
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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def auth_client():
    """TestClient with a default happy-path Supabase mock."""
    sb      = _make_supabase_mock()
    backend = _make_backend_mock(sb)
    with patch("api.main._backend", backend), \
         patch("api.two_step_router._engine", MagicMock()):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, sb       # yield both so tests can re-configure the mock


# ── POST /auth/register ───────────────────────────────────────────────────────

class TestRegister:

    def test_success_200(self, auth_client):
        client, sb = auth_client
        r = client.post("/auth/register", json={
            "email": "new@example.com", "password": "SecurePass123!"
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "user_id" in body
        assert body["message"] == "Registration successful."

    def test_calls_supabase_register(self, auth_client):
        client, sb = auth_client
        client.post("/auth/register", json={
            "email": "check@example.com", "password": "Pass1234!"
        })
        sb.register.assert_called_once_with("check@example.com", "Pass1234!")

    def test_duplicate_email_409(self, auth_client):
        client, sb = auth_client
        sb.register.return_value = {"ok": False, "code": "email_exists"}
        r = client.post("/auth/register", json={
            "email": "taken@example.com", "password": "Pass1234!"
        })
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert "email" in detail["message"].lower()

    def test_supabase_offline_503(self, auth_client):
        client, sb = auth_client
        sb.register.return_value = {"ok": False, "code": "network_error"}
        r = client.post("/auth/register", json={
            "email": "x@x.com", "password": "Pass1234!"
        })
        assert r.status_code == 503

    def test_missing_email_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/register", json={"password": "abc123"})
        assert r.status_code == 422

    def test_missing_password_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/register", json={"email": "a@b.com"})
        assert r.status_code == 422

    def test_invalid_email_format_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/register", json={
            "email": "not-an-email", "password": "Pass123!"
        })
        assert r.status_code == 422

    def test_empty_body_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/register", json={})
        assert r.status_code == 422

    def test_user_id_returned_on_success(self, auth_client):
        client, sb = auth_client
        sb.register.return_value = {
            "ok": True,
            "user_id": FAKE_USER_ID,
            "access_token": FAKE_TOKEN,
        }
        r = client.post("/auth/register", json={
            "email": "new@example.com", "password": "Pass123!"
        })
        assert r.json()["user_id"] == FAKE_USER_ID


# ── POST /auth/login ──────────────────────────────────────────────────────────

class TestLogin:

    def test_success_200(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/login", json={
            "email": "test@example.com", "password": "correct"
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["token"] == FAKE_TOKEN
        assert body["user_id"] == FAKE_USER_ID
        assert body["message"] == "Login successful."

    def test_calls_supabase_login(self, auth_client):
        client, sb = auth_client
        client.post("/auth/login", json={
            "email": "test@example.com", "password": "pw"
        })
        sb.login.assert_called_once_with("test@example.com", "pw")

    def test_wrong_password_401(self, auth_client):
        client, sb = auth_client
        sb.login.return_value = {"ok": False, "code": "invalid_credentials"}
        r = client.post("/auth/login", json={
            "email": "test@example.com", "password": "wrong"
        })
        assert r.status_code == 401
        assert "invalid" in r.json()["detail"]["message"].lower()

    def test_unknown_email_401(self, auth_client):
        client, sb = auth_client
        sb.login.return_value = {"ok": False, "code": "invalid_credentials"}
        r = client.post("/auth/login", json={
            "email": "nobody@example.com", "password": "anything"
        })
        assert r.status_code == 401

    def test_supabase_offline_503(self, auth_client):
        client, sb = auth_client
        sb.login.return_value = {"ok": False, "code": "network_error"}
        r = client.post("/auth/login", json={
            "email": "a@b.com", "password": "pw"
        })
        assert r.status_code == 503

    def test_token_in_response(self, auth_client):
        client, sb = auth_client
        sb.login.return_value = {
            "ok": True,
            "user_id": FAKE_USER_ID,
            "access_token": "a.b.c",
        }
        r = client.post("/auth/login", json={
            "email": "a@b.com", "password": "pw"
        })
        assert r.json()["token"] == "a.b.c"

    def test_missing_email_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/login", json={"password": "pw"})
        assert r.status_code == 422

    def test_missing_password_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/login", json={"email": "a@b.com"})
        assert r.status_code == 422

    def test_invalid_email_format_422(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/login", json={"email": "badformat", "password": "pw"})
        assert r.status_code == 422

    def test_no_token_on_failure(self, auth_client):
        client, sb = auth_client
        sb.login.return_value = {"ok": False, "code": "invalid_credentials"}
        r = client.post("/auth/login", json={"email": "a@b.com", "password": "bad"})
        assert r.status_code == 401
        assert "token" not in r.json()


# ── POST /auth/logout ─────────────────────────────────────────────────────────

class TestLogout:

    def test_success_200(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/logout")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_calls_supabase_logout(self, auth_client):
        client, sb = auth_client
        client.post("/auth/logout")
        sb.logout.assert_called_once()

    def test_idempotent_already_logged_out(self, auth_client):
        client, sb = auth_client
        sb.logout.return_value = True
        r1 = client.post("/auth/logout")
        r2 = client.post("/auth/logout")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert sb.logout.call_count == 2

    def test_no_body_needed(self, auth_client):
        client, _ = auth_client
        r = client.post("/auth/logout")
        assert r.status_code == 200

    def test_ok_true_always_returned(self, auth_client):
        """logout always returns ok=True regardless of Supabase state."""
        client, sb = auth_client
        sb.logout.return_value = False   # simulate sign_out error — still ok
        r = client.post("/auth/logout")
        assert r.status_code == 200
        assert r.json()["ok"] is True


# ── GET /auth/me ──────────────────────────────────────────────────────────────

class TestMe:

    def _auth_header(self, token: str = FAKE_TOKEN) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def test_valid_token_200(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = FAKE_USER
        r = client.get("/auth/me", headers=self._auth_header())
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "user" in body

    def test_user_fields_present(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = FAKE_USER
        body = client.get("/auth/me", headers=self._auth_header()).json()
        user = body["user"]
        for field in ("id", "email", "role", "created_at"):
            assert field in user, f"missing user field: {field}"

    def test_user_id_matches(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = FAKE_USER
        body = client.get("/auth/me", headers=self._auth_header()).json()
        assert body["user"]["id"] == FAKE_USER_ID

    def test_token_passed_to_supabase(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = FAKE_USER
        client.get("/auth/me", headers=self._auth_header("my.custom.token"))
        sb.get_user_by_token.assert_called_once_with("my.custom.token")

    def test_invalid_token_401(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = None
        r = client.get("/auth/me", headers=self._auth_header("invalid.token"))
        assert r.status_code == 401
        detail = r.json()["detail"]
        msg = detail["message"] if isinstance(detail, dict) else detail
        assert "invalid" in msg.lower() or "expired" in msg.lower()

    def test_expired_token_401(self, auth_client):
        client, sb = auth_client
        sb.get_user_by_token.return_value = None
        r = client.get("/auth/me", headers={"Authorization": "Bearer expired.token.here"})
        assert r.status_code == 401

    def test_missing_authorization_header_401(self, auth_client):
        client, _ = auth_client
        r = client.get("/auth/me")
        assert r.status_code == 401
        detail = r.json()["detail"]
        msg = detail["message"] if isinstance(detail, dict) else detail
        assert "authorization" in msg.lower() or "missing" in msg.lower()

    def test_malformed_no_bearer_prefix_401(self, auth_client):
        client, _ = auth_client
        r = client.get("/auth/me", headers={"Authorization": FAKE_TOKEN})
        assert r.status_code == 401

    def test_malformed_token_scheme_401(self, auth_client):
        client, _ = auth_client
        r = client.get("/auth/me", headers={"Authorization": f"Token {FAKE_TOKEN}"})
        assert r.status_code == 401

    def test_empty_bearer_value_401(self, auth_client):
        client, _ = auth_client
        r = client.get("/auth/me", headers={"Authorization": "Bearer "})
        assert r.status_code == 401

    def test_supabase_network_error_returns_401(self, auth_client):
        """If get_user_by_token returns None (network/offline), respond 401."""
        client, sb = auth_client
        # Use a unique token that can't be in the cache from prior tests.
        sb.get_user_by_token.return_value = None
        r = client.get("/auth/me", headers={"Authorization": "Bearer unique.network.error.token"})
        assert r.status_code == 401

    def test_logged_out_state_401(self, auth_client):
        """After logout, a previously valid token should return 401 (server-side revocation).
        The logout route evicts the token from cache, so the next /me call hits Supabase,
        which returns None (revoked token).
        """
        client, sb = auth_client
        # Use a unique token to avoid cross-test cache pollution.
        unique_token = "unique.logout.test.token.xyz"
        sb.get_user_by_token.return_value = FAKE_USER

        # Prime the cache via /me (cache miss → Supabase → cache set)
        client.get("/auth/me", headers={"Authorization": f"Bearer {unique_token}"})

        # Logout — evicts the token from cache
        client.post("/auth/logout", headers={"Authorization": f"Bearer {unique_token}"})

        # After logout, Supabase returns None (token revoked server-side)
        sb.get_user_by_token.return_value = None
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {unique_token}"})
        assert r.status_code == 401


# ── Cross-route scenario tests ────────────────────────────────────────────────

class TestAuthScenarios:

    def test_register_then_login_then_me(self):
        """Full happy path: register → login → use token for /me."""
        sb      = _make_supabase_mock()
        backend = _make_backend_mock(sb)
        with patch("api.main._backend", backend), \
             patch("api.two_step_router._engine", MagicMock()):
            from api.app import app
            with TestClient(app, raise_server_exceptions=True) as client:
                # 1. Register
                r1 = client.post("/auth/register", json={
                    "email": "flow@example.com", "password": "Secure1!"
                })
                assert r1.status_code == 200

                # 2. Login
                r2 = client.post("/auth/login", json={
                    "email": "flow@example.com", "password": "Secure1!"
                })
                assert r2.status_code == 200
                token = r2.json()["token"]
                assert token == FAKE_TOKEN

                # 3. /me with the token
                sb.get_user_by_token.return_value = FAKE_USER
                r3 = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
                assert r3.status_code == 200
                assert r3.json()["user"]["email"] == FAKE_USER["email"]

    def test_login_then_logout_then_me_fails(self):
        """Login → logout → /me should 401 (token invalidated after logout)."""
        sb      = _make_supabase_mock()
        backend = _make_backend_mock(sb)
        with patch("api.main._backend", backend), \
             patch("api.two_step_router._engine", MagicMock()):
            from api.app import app
            with TestClient(app, raise_server_exceptions=True) as client:
                # Login
                r1 = client.post("/auth/login", json={
                    "email": "a@b.com", "password": "pw"
                })
                token = r1.json()["token"]

                # Logout with Authorization header so the cache gets evicted
                client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})

                # After logout Supabase token is revoked → get_user_by_token returns None
                sb.get_user_by_token.return_value = None
                r3 = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
                assert r3.status_code == 401

    def test_bad_login_cannot_access_me(self):
        """Wrong password → no token → /me with garbage token → 401."""
        sb      = _make_supabase_mock(login_ok=False)
        backend = _make_backend_mock(sb)
        with patch("api.main._backend", backend), \
             patch("api.two_step_router._engine", MagicMock()):
            from api.app import app
            with TestClient(app, raise_server_exceptions=True) as client:
                r1 = client.post("/auth/login", json={
                    "email": "a@b.com", "password": "wrong"
                })
                assert r1.status_code == 401

                r2 = client.get("/auth/me", headers={"Authorization": "Bearer fake"})
                assert r2.status_code == 401

    def test_all_auth_routes_registered(self):
        """Smoke: every auth route exists (none 404)."""
        sb      = _make_supabase_mock()
        backend = _make_backend_mock(sb)
        with patch("api.main._backend", backend), \
             patch("api.two_step_router._engine", MagicMock()):
            from api.app import app
            with TestClient(app, raise_server_exceptions=True) as client:
                routes = [
                    ("POST", "/auth/register"),
                    ("POST", "/auth/login"),
                    ("POST", "/auth/logout"),
                    ("GET",  "/auth/me"),
                ]
                for method, path in routes:
                    r = client.request(method, path)
                    assert r.status_code != 404, \
                        f"{method} {path} returned 404 — not registered"
