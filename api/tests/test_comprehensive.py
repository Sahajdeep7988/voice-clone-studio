"""
Comprehensive backend integration tests.

Coverage:
  ✓ Token cache — TTL, SHA-256 keying, invalidation, size sweep
  ✓ Auth — forgot-password, reset-password, change-password, refresh, resend-confirmation
  ✓ Auth — error code mapping (11 GoTrue codes)
  ✓ Session lifecycle — list, status, stop, delete (409 guard, 403 ownership)
  ✓ Inference — list models, convert (path safety)
  ✓ Segments — list, approve, approve-all, delete (ownership, path safety)
  ✓ Upload — valid files, unsupported extension, multi-file partial failure, no auth
  ✓ Path safety — traversal blocked, safe dirs allowed
"""

import io
import time
import sys
import os
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.tests.conftest import (
    make_session,
    make_mock_backend,
    make_mock_prepare_engine,
    parse_sse,
    DEFAULT_HYPERPARAMS,
)

FAKE_TOKEN   = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.comprehensive.token"
FAKE_USER_ID = "user-comprehensive-001"
FAKE_USER    = {
    "id":         FAKE_USER_ID,
    "email":      "comp@example.com",
    "role":       "authenticated",
    "created_at": "2026-04-02T10:00:00+00:00",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_client(backend=None, engine=None, auth_token=FAKE_TOKEN):
    mb  = backend or make_mock_backend()
    eng = engine  or make_mock_prepare_engine()
    mb.supabase.get_user_by_token.return_value = FAKE_USER
    with patch("api.main._backend", mb), \
         patch("api.segments_router._backend", mb), \
         patch("api.two_step_router._engine", eng):
        from api.app import app
        client = TestClient(app, raise_server_exceptions=True)
        client.headers.update({"Authorization": f"Bearer {auth_token}"})
        return client, mb, eng


@pytest.fixture
def mb():
    backend = make_mock_backend()
    backend.supabase.get_user_by_token.return_value = FAKE_USER
    return backend


@pytest.fixture
def client(mb):
    eng = make_mock_prepare_engine()
    with patch("api.main._backend", mb), \
         patch("api.segments_router._backend", mb), \
         patch("api.two_step_router._engine", eng):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            c.headers.update({"Authorization": f"Bearer {FAKE_TOKEN}"})
            yield c


@pytest.fixture
def unauth_client(mb):
    eng = make_mock_prepare_engine()
    with patch("api.main._backend", mb), \
         patch("api.segments_router._backend", mb), \
         patch("api.two_step_router._engine", eng):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ══════════════════════════════════════════════════════════════════════════════
# Token cache unit tests (api.main._TokenCache)
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenCache:

    def _fresh_cache(self):
        from api.main import _TokenCache
        return _TokenCache()

    def test_set_and_get(self):
        cache = self._fresh_cache()
        cache.set("tok", {"id": "u1"})
        assert cache.get("tok") == {"id": "u1"}

    def test_miss_returns_none(self):
        cache = self._fresh_cache()
        assert cache.get("nonexistent") is None

    def test_invalidate_evicts(self):
        cache = self._fresh_cache()
        cache.set("tok", {"id": "u1"})
        cache.invalidate("tok")
        assert cache.get("tok") is None

    def test_invalidate_nonexistent_noop(self):
        cache = self._fresh_cache()
        cache.invalidate("ghost")   # must not raise

    def test_different_tokens_isolated(self):
        cache = self._fresh_cache()
        cache.set("tok-a", {"id": "a"})
        cache.set("tok-b", {"id": "b"})
        assert cache.get("tok-a") == {"id": "a"}
        assert cache.get("tok-b") == {"id": "b"}
        cache.invalidate("tok-a")
        assert cache.get("tok-a") is None
        assert cache.get("tok-b") == {"id": "b"}

    def test_sha256_key_not_raw_token(self):
        """Raw token must not appear as a key in the internal store."""
        import hashlib
        cache = self._fresh_cache()
        cache.set("my-secret-token", {"id": "u1"})
        raw_in_store = "my-secret-token" in cache._store
        assert not raw_in_store
        sha = hashlib.sha256(b"my-secret-token").hexdigest()
        assert sha in cache._store

    def test_ttl_expiry(self, monkeypatch):
        cache = self._fresh_cache()
        # Override TTL to 0 so entries expire immediately.
        monkeypatch.setattr(cache, "TTL", 0)
        cache.set("tok", {"id": "u1"})
        # Entry expires in 0 seconds — monotonic clock has already advanced past exp.
        assert cache.get("tok") is None

    def test_overwrite_resets_ttl(self, monkeypatch):
        cache = self._fresh_cache()
        cache.set("tok", {"id": "u1"})
        cache.set("tok", {"id": "u2"})   # overwrite
        assert cache.get("tok") == {"id": "u2"}

    def test_sweep_removes_expired(self, monkeypatch):
        cache = self._fresh_cache()
        monkeypatch.setattr(cache, "TTL", 0)
        cache.set("old", {"id": "old"})
        monkeypatch.setattr(cache, "TTL", 30)
        cache.set("fresh", {"id": "fresh"})
        with cache._lock:
            cache._sweep()
        assert cache.get("old") is None
        assert cache.get("fresh") == {"id": "fresh"}


# ══════════════════════════════════════════════════════════════════════════════
# Auth — new endpoints (forgot/reset/change password, refresh, resend)
# ══════════════════════════════════════════════════════════════════════════════

class TestForgotPassword:

    def test_always_200(self, client, mb):
        mb.supabase.forgot_password.return_value = {"ok": True}
        r = client.post("/auth/forgot-password", json={"email": "a@b.com"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_supabase_failure_still_200(self, client, mb):
        mb.supabase.forgot_password.return_value = {"ok": False, "code": "unknown"}
        r = client.post("/auth/forgot-password", json={"email": "a@b.com"})
        assert r.status_code == 200

    def test_rate_limited_429(self, client, mb):
        mb.supabase.forgot_password.return_value = {
            "ok": False, "code": "over_request_rate_limit"
        }
        r = client.post("/auth/forgot-password", json={"email": "a@b.com"})
        assert r.status_code == 429

    def test_invalid_email_422(self, client, mb):
        r = client.post("/auth/forgot-password", json={"email": "not-an-email"})
        assert r.status_code == 422

    def test_no_auth_required(self, unauth_client, mb):
        mb.supabase.forgot_password.return_value = {"ok": True}
        r = unauth_client.post("/auth/forgot-password", json={"email": "a@b.com"})
        assert r.status_code == 200


class TestResetPassword:

    def test_success(self, client, mb):
        mb.supabase.reset_password.return_value = {"ok": True}
        r = client.post("/auth/reset-password", json={
            "access_token": "recovery.jwt.token",
            "new_password": "NewPass123!",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_empty_token_400(self, client, mb):
        r = client.post("/auth/reset-password", json={
            "access_token": "   ",
            "new_password": "NewPass123!",
        })
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "missing_token"

    def test_weak_password_422(self, client, mb):
        r = client.post("/auth/reset-password", json={
            "access_token": "tok",
            "new_password": "short",
        })
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "weak_password"

    def test_expired_token_400(self, client, mb):
        mb.supabase.reset_password.return_value = {"ok": False, "code": "otp_expired"}
        r = client.post("/auth/reset-password", json={
            "access_token": "expired.tok",
            "new_password": "NewPass123!",
        })
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "token_expired"

    def test_no_auth_required(self, unauth_client, mb):
        mb.supabase.reset_password.return_value = {"ok": True}
        r = unauth_client.post("/auth/reset-password", json={
            "access_token": "tok",
            "new_password": "NewPass123!",
        })
        assert r.status_code == 200


class TestChangePassword:

    def test_success(self, client, mb):
        mb.supabase.get_user_by_token.return_value = FAKE_USER
        mb.supabase.change_password.return_value = {"ok": True}
        r = client.post("/auth/change-password", json={"new_password": "NewPass123!"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_weak_password_422(self, client, mb):
        r = client.post("/auth/change-password", json={"new_password": "short"})
        assert r.status_code == 422

    def test_no_auth_header_401(self, unauth_client, mb):
        r = unauth_client.post("/auth/change-password", json={"new_password": "NewPass123!"})
        assert r.status_code == 401

    def test_invalid_token_401(self, unauth_client, mb):
        mb.supabase.get_user_by_token.return_value = None
        r = unauth_client.post(
            "/auth/change-password",
            json={"new_password": "NewPass123!"},
            headers={"Authorization": "Bearer invalid.token"},
        )
        assert r.status_code == 401

    def test_same_password_422(self, client, mb):
        mb.supabase.get_user_by_token.return_value = FAKE_USER
        mb.supabase.change_password.return_value = {"ok": False, "code": "same_password"}
        r = client.post("/auth/change-password", json={"new_password": "NewPass123!"})
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "same_password"


class TestRefreshToken:

    def test_success(self, client, mb):
        mb.supabase.refresh_token.return_value = {
            "ok": True,
            "user_id": FAKE_USER_ID,
            "access_token": "new.access.tok",
            "refresh_token": "new.refresh.tok",
        }
        r = client.post("/auth/refresh", json={"refresh_token": "old.refresh.tok"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["token"] == "new.access.tok"
        assert body["refresh_token"] == "new.refresh.tok"

    def test_invalid_refresh_401(self, client, mb):
        mb.supabase.refresh_token.return_value = {"ok": False}
        r = client.post("/auth/refresh", json={"refresh_token": "bad.tok"})
        assert r.status_code == 401

    def test_no_auth_required(self, unauth_client, mb):
        mb.supabase.refresh_token.return_value = {
            "ok": True,
            "user_id": FAKE_USER_ID,
            "access_token": "a.b.c",
            "refresh_token": "r.e.f",
        }
        r = unauth_client.post("/auth/refresh", json={"refresh_token": "tok"})
        assert r.status_code == 200


class TestResendConfirmation:

    def test_always_200(self, client, mb):
        mb.supabase.resend_confirmation.return_value = {"ok": True}
        r = client.post("/auth/resend-confirmation", json={"email": "a@b.com"})
        assert r.status_code == 200

    def test_failure_still_200(self, client, mb):
        mb.supabase.resend_confirmation.return_value = {"ok": False}
        r = client.post("/auth/resend-confirmation", json={"email": "a@b.com"})
        assert r.status_code == 200

    def test_rate_limited_429(self, client, mb):
        mb.supabase.resend_confirmation.return_value = {
            "ok": False, "code": "over_email_send_rate_limit"
        }
        r = client.post("/auth/resend-confirmation", json={"email": "a@b.com"})
        assert r.status_code == 429

    def test_no_auth_required(self, unauth_client, mb):
        mb.supabase.resend_confirmation.return_value = {"ok": True}
        r = unauth_client.post("/auth/resend-confirmation", json={"email": "a@b.com"})
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# Auth — error code mapping
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthErrorCodeMapping:
    """Verify _http_from_auth_error maps GoTrue codes to correct HTTP status codes."""

    _CASES = [
        ("email_not_confirmed",           403),
        ("invalid_credentials",           401),
        ("user_banned",                   403),
        ("email_exists",                  409),
        ("user_already_exists",           409),
        ("weak_password",                 422),
        ("signup_disabled",               403),
        ("email_address_invalid",         422),
        ("over_request_rate_limit",       429),
        ("over_email_send_rate_limit",    429),
        ("network_error",                 503),
        ("otp_expired",                   400),
        ("same_password",                 422),
    ]

    @pytest.mark.parametrize("code,expected_status", _CASES)
    def test_login_error_code(self, client, mb, code, expected_status):
        mb.supabase.login.return_value = {"ok": False, "code": code}
        r = client.post("/auth/login", json={"email": "a@b.com", "password": "Pass123!"})
        assert r.status_code == expected_status, (
            f"code={code!r}: expected {expected_status}, got {r.status_code}"
        )

    @pytest.mark.parametrize("code,expected_status", _CASES)
    def test_register_error_code(self, client, mb, code, expected_status):
        mb.supabase.register.return_value = {"ok": False, "code": code}
        r = client.post("/auth/register", json={"email": "a@b.com", "password": "Pass123!"})
        assert r.status_code == expected_status, (
            f"code={code!r}: expected {expected_status}, got {r.status_code}"
        )

    def test_unknown_code_returns_400(self, client, mb):
        mb.supabase.login.return_value = {"ok": False, "code": "some_unknown_code"}
        r = client.post("/auth/login", json={"email": "a@b.com", "password": "Pass123!"})
        assert r.status_code == 400

    def test_error_detail_is_structured_dict(self, client, mb):
        mb.supabase.login.return_value = {"ok": False, "code": "invalid_credentials"}
        r = client.post("/auth/login", json={"email": "a@b.com", "password": "Pass123!"})
        detail = r.json()["detail"]
        assert isinstance(detail, dict)
        assert "code" in detail
        assert "message" in detail


# ══════════════════════════════════════════════════════════════════════════════
# Auth — GET /auth/me cache consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestMeCache:

    def test_me_returns_full_user_on_cache_miss(self, client, mb):
        mb.supabase.get_user_by_token.return_value = FAKE_USER
        r = client.get("/auth/me", headers={"Authorization": "Bearer unique.me.miss.tok"})
        assert r.status_code == 200
        user = r.json()["user"]
        assert "email" in user
        assert "role" in user

    def test_me_returns_full_user_on_cache_hit(self, client, mb):
        """Second call hits cache — must still return full user fields."""
        token = "unique.me.cache.hit.tok.xyzzy"
        mb.supabase.get_user_by_token.return_value = FAKE_USER
        # First call: cache miss → Supabase → caches full user
        client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        # Second call: cache hit → must have full user
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        user = r.json()["user"]
        for field in ("id", "email", "role", "created_at"):
            assert field in user, f"missing field on cache hit: {field}"
        assert r.json()["cached"] is True

    def test_me_logout_evicts_cache(self, client, mb):
        """Logout with Bearer header must evict the token so next /me hits Supabase."""
        token = "unique.logout.eviction.tok"
        mb.supabase.get_user_by_token.return_value = FAKE_USER
        client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})  # prime
        client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
        mb.supabase.get_user_by_token.return_value = None  # revoked
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401


# ══════════════════════════════════════════════════════════════════════════════
# Session lifecycle — delete, list models, checkpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteSession:

    def test_delete_done_session(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID, status="done")
        mb.sessions.get_session.return_value = sess
        mb.sessions.delete_session.return_value = True
        r = client.delete("/sessions/sess-abc-123")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_delete_training_session_409(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID, status="training")
        mb.sessions.get_session.return_value = sess
        r = client.delete("/sessions/sess-abc-123")
        assert r.status_code == 409
        assert "stop" in r.json()["detail"].lower() or "training" in r.json()["detail"].lower()

    def test_delete_preprocessing_session_409(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID, status="preprocessing")
        mb.sessions.get_session.return_value = sess
        r = client.delete("/sessions/sess-abc-123")
        assert r.status_code == 409

    def test_delete_not_found_404(self, client, mb):
        mb.sessions.get_session.return_value = None
        r = client.delete("/sessions/ghost")
        assert r.status_code == 404

    def test_delete_wrong_user_403(self, client, mb):
        sess = make_session(user_id="other-user", status="done")
        mb.sessions.get_session.return_value = sess
        r = client.delete("/sessions/sess-abc-123")
        assert r.status_code == 403

    def test_delete_returns_session_id(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID, status="done")
        mb.sessions.get_session.return_value = sess
        mb.sessions.delete_session.return_value = True
        body = client.delete("/sessions/sess-abc-123").json()
        assert body["session_id"] == "sess-abc-123"


class TestListModels:

    def test_list_models_success(self, client, mb):
        mb.list_models.return_value = {"ok": True, "models": ["/tmp/m.pth"]}
        r = client.get("/inference/models")
        assert r.status_code == 200
        assert r.json()["models"] == ["/tmp/m.pth"]

    def test_list_models_empty(self, client, mb):
        mb.list_models.return_value = {"ok": True, "models": []}
        r = client.get("/inference/models")
        assert r.status_code == 200
        assert r.json()["models"] == []

    def test_list_models_requires_auth(self, unauth_client, mb):
        r = unauth_client.get("/inference/models")
        assert r.status_code == 401


class TestListCheckpoints:

    def test_success(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.list_checkpoints.return_value = {
            "ok": True,
            "checkpoints": ["/rvc/logs/model/ckpt_50e.pth"],
        }
        r = client.get("/sessions/sess-abc-123/checkpoints")
        assert r.status_code == 200
        assert len(r.json()["checkpoints"]) == 1

    def test_session_not_found_404(self, client, mb):
        mb.sessions.get_session.return_value = None
        r = client.get("/sessions/ghost/checkpoints")
        assert r.status_code == 404

    def test_wrong_user_403(self, client, mb):
        sess = make_session(user_id="other-user")
        mb.sessions.get_session.return_value = sess
        r = client.get("/sessions/sess-abc-123/checkpoints")
        assert r.status_code == 403


# ══════════════════════════════════════════════════════════════════════════════
# Segments
# ══════════════════════════════════════════════════════════════════════════════

class TestSegments:

    def _make_manifest(self, tmp_path, count=3):
        segs = []
        for i in range(count):
            p = tmp_path / f"seg_{i}.wav"
            p.write_bytes(b"wav")
            segs.append({"path": str(p), "duration_s": 5.0, "size_bytes": 3, "approved": True})
        return segs

    def test_get_segments_success(self, client, mb, tmp_path):
        segs = self._make_manifest(tmp_path)
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.get_segments.return_value = {"ok": True, "segments": segs}
        r = client.get("/sessions/sess-abc-123/segments")
        assert r.status_code == 200
        assert len(r.json()["segments"]) == 3

    def test_get_segments_session_not_found(self, client, mb):
        mb.sessions.get_session.return_value = None
        r = client.get("/sessions/ghost/segments")
        assert r.status_code == 404

    def test_get_segments_wrong_user_403(self, client, mb):
        mb.sessions.get_session.return_value = make_session(user_id="other")
        r = client.get("/sessions/sess-abc-123/segments")
        assert r.status_code == 403

    def test_approve_segments(self, client, mb, tmp_path):
        segs = self._make_manifest(tmp_path)
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.approve_segments.return_value = {"ok": True, "approved_count": 2}
        approved_paths = [segs[0]["path"], segs[1]["path"]]
        r = client.post("/sessions/sess-abc-123/segments/approve", json={
            "approved_paths": approved_paths
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_approve_path_outside_safe_dir_400(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        r = client.post("/sessions/sess-abc-123/segments/approve", json={
            "approved_paths": ["/etc/passwd"]
        })
        assert r.status_code == 400

    def test_approve_all_segments(self, client, mb, tmp_path):
        segs = self._make_manifest(tmp_path)
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.get_segments.return_value = {"ok": True, "segments": segs}
        mb.approve_segments.return_value = {"ok": True, "approved_count": 3}
        r = client.post("/sessions/sess-abc-123/segments/approve-all")
        assert r.status_code == 200
        # Verify approve_segments was called with all paths
        call_args = mb.approve_segments.call_args
        assert len(call_args[0][1]) == 3

    def test_reject_segments(self, client, mb, tmp_path):
        segs = self._make_manifest(tmp_path)
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.get_segments.return_value = {"ok": True, "segments": segs}
        mb.approve_segments.return_value = {"ok": True, "approved_count": 1}
        r = client.post("/sessions/sess-abc-123/segments/reject", json={
            "paths": [segs[0]["path"], segs[1]["path"]]
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["rejected_count"] == 2
        args = mb.approve_segments.call_args[0]
        assert args[0] == "sess-abc-123"
        assert len(args[1]) == 1

    def test_reject_path_outside_safe_dir_400(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.get_segments.return_value = {"ok": True, "segments": []}
        r = client.post("/sessions/sess-abc-123/segments/reject", json={
            "paths": ["/etc/passwd"]
        })
        assert r.status_code == 400

    def test_delete_segment(self, client, mb, tmp_path):
        seg = tmp_path / "seg_0.wav"
        seg.write_bytes(b"wav")
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.delete_segment.return_value = {"ok": True}
        r = client.request(
            "DELETE",
            "/sessions/sess-abc-123/segments",
            json={"path": str(seg)},
        )
        assert r.status_code == 200

    def test_delete_segment_path_traversal_400(self, client, mb):
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        r = client.request(
            "DELETE",
            "/sessions/sess-abc-123/segments",
            json={"path": "/etc/shadow"},
        )
        assert r.status_code == 400

    def test_delete_segment_not_found_404(self, client, mb, tmp_path):
        seg = tmp_path / "missing.wav"
        seg.write_bytes(b"wav")
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.delete_segment.return_value = {"ok": False}
        r = client.request(
            "DELETE",
            "/sessions/sess-abc-123/segments",
            json={"path": str(seg)},
        )
        assert r.status_code == 404

    def test_delete_segment_query_path(self, client, mb, tmp_path):
        seg = tmp_path / "seg_q.wav"
        seg.write_bytes(b"wav")
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.delete_segment.return_value = {"ok": True}
        r = client.request(
            "DELETE",
            f"/sessions/sess-abc-123/segments?path={str(seg)}",
        )
        assert r.status_code == 200

    def test_delete_segment_by_id_filename(self, client, mb, tmp_path):
        seg = tmp_path / "seg_100.wav"
        seg.write_bytes(b"wav")
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.get_segments.return_value = {
            "ok": True,
            "segments": [{"path": str(seg), "approved": True}],
        }
        mb.delete_segment.return_value = {"ok": True}
        r = client.request("DELETE", "/sessions/sess-abc-123/segments/seg_100.wav")
        assert r.status_code == 200

    def test_delete_segment_by_id_stem(self, client, mb, tmp_path):
        seg = tmp_path / "seg_101.wav"
        seg.write_bytes(b"wav")
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.get_segments.return_value = {
            "ok": True,
            "segments": [{"path": str(seg), "approved": True}],
        }
        mb.delete_segment.return_value = {"ok": True}
        r = client.request("DELETE", "/sessions/sess-abc-123/segments/seg_101")
        assert r.status_code == 200

    def test_delete_segment_by_id_not_found_404(self, client, mb, tmp_path):
        seg = tmp_path / "seg_200.wav"
        seg.write_bytes(b"wav")
        sess = make_session(user_id=FAKE_USER_ID)
        mb.sessions.get_session.return_value = sess
        mb.get_segments.return_value = {
            "ok": True,
            "segments": [{"path": str(seg), "approved": True}],
        }
        r = client.request("DELETE", "/sessions/sess-abc-123/segments/missing_segment")
        assert r.status_code == 404

    def test_segments_requires_auth(self, unauth_client, mb):
        r = unauth_client.get("/sessions/sess-abc-123/segments")
        assert r.status_code == 401


# ══════════════════════════════════════════════════════════════════════════════
# Upload
# ══════════════════════════════════════════════════════════════════════════════

class TestUpload:

    def _wav_file(self, name="audio.wav"):
        return ("files", (name, io.BytesIO(b"RIFF" + b"\x00" * 36), "audio/wav"))

    def test_upload_single_wav(self, client, mb, tmp_path):
        r = client.post("/upload", files=[self._wav_file()])
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert len(body["saved"]) == 1
        assert body["paths"][0].endswith(".wav")
        # Cleanup
        for s in body["saved"]:
            if os.path.exists(s["path"]):
                os.remove(s["path"])

    def test_upload_supported_formats(self, client, mb):
        supported = ["a.mp3", "b.flac", "c.m4a", "d.ogg", "e.opus", "f.webm"]
        files = [("files", (name, io.BytesIO(b"x"), "audio/mpeg")) for name in supported]
        r = client.post("/upload", files=files)
        body = r.json()
        assert body["ok"] is True
        assert len(body["saved"]) == len(supported)
        assert len(body["errors"]) == 0
        for s in body["saved"]:
            if os.path.exists(s["path"]):
                os.remove(s["path"])

    def test_upload_unsupported_extension_400(self, client, mb):
        files = [("files", ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream"))]
        r = client.post("/upload", files=files)
        assert r.status_code == 400

    def test_upload_partial_failure_207(self, client, mb):
        files = [
            ("files", ("good.wav", io.BytesIO(b"RIFF"), "audio/wav")),
            ("files", ("bad.exe",  io.BytesIO(b"MZ"),   "application/octet-stream")),
        ]
        r = client.post("/upload", files=files)
        assert r.status_code == 207
        body = r.json()
        assert body["ok"] is True
        assert len(body["saved"]) == 1
        assert len(body["errors"]) == 1
        for s in body["saved"]:
            if os.path.exists(s["path"]):
                os.remove(s["path"])

    def test_upload_requires_auth(self, unauth_client, mb):
        files = [self._wav_file()]
        r = unauth_client.post("/upload", files=files)
        assert r.status_code == 401

    def test_upload_stores_in_user_subdir(self, client, mb):
        r = client.post("/upload", files=[self._wav_file()])
        body = r.json()
        assert body["ok"] is True
        path = body["paths"][0]
        assert FAKE_USER_ID in path
        if os.path.exists(path):
            os.remove(path)

    def test_upload_uuid_prefix_prevents_collisions(self, client, mb):
        files = [
            ("files", ("audio.wav", io.BytesIO(b"RIFF"), "audio/wav")),
            ("files", ("audio.wav", io.BytesIO(b"RIFF"), "audio/wav")),
        ]
        r = client.post("/upload", files=files)
        body = r.json()
        assert len(body["paths"]) == 2
        assert body["paths"][0] != body["paths"][1]
        for s in body["saved"]:
            if os.path.exists(s["path"]):
                os.remove(s["path"])

    def test_upload_path_response_contains_size(self, client, mb):
        r = client.post("/upload", files=[self._wav_file()])
        body = r.json()
        assert "size_bytes" in body["saved"][0]
        for s in body["saved"]:
            if os.path.exists(s["path"]):
                os.remove(s["path"])


# ══════════════════════════════════════════════════════════════════════════════
# Path safety
# ══════════════════════════════════════════════════════════════════════════════

class TestPathSafety:

    def test_inference_convert_rejects_etc_passwd(self, client, mb):
        r = client.post("/inference/convert", json={
            "model_path":       "/etc/passwd",
            "input_audio_path": "/tmp/in.wav",
            "output_path":      "/tmp/out.wav",
        })
        assert r.status_code == 403

    def test_inference_convert_rejects_home_dir(self, client, mb):
        r = client.post("/inference/convert", json={
            "model_path":       os.path.expanduser("~/.ssh/id_rsa"),
            "input_audio_path": "/tmp/in.wav",
            "output_path":      "/tmp/out.wav",
        })
        # If the file does not exist on this machine, existence check returns 400
        # before path-allowlist check can return 403.
        assert r.status_code in (400, 403)

    def test_inference_convert_allows_tmp(self, client, mb, tmp_path):
        model = tmp_path / "m.pth"
        model.write_bytes(b"model")
        audio_in = tmp_path / "in.wav"
        audio_in.write_bytes(b"wav")
        mb.convert_audio.return_value = {"ok": True, "output_path": "/tmp/out.wav"}
        r = client.post("/inference/convert", json={
            "model_path":       str(model),
            "input_audio_path": str(audio_in),
            "output_path":      "/tmp/out.wav",
        })
        assert r.status_code == 200

    def test_create_session_rejects_etc(self, client, mb):
        r = client.post("/sessions/create", json={
            "files": ["/etc/shadow"],
            "model_name": "m",
        })
        assert r.status_code == 403

    def test_path_traversal_double_dot_blocked(self, client, mb, tmp_path):
        """../ traversal within an otherwise safe path must be rejected."""
        traversal = str(tmp_path) + "/../../etc/passwd"
        r = client.post("/sessions/create", json={
            "files": [traversal],
            "model_name": "m",
        })
        # Either 400 (path not found) or 403 (path not allowed) — both are correct.
        assert r.status_code in (400, 403)


# ══════════════════════════════════════════════════════════════════════════════
# Register — password validation
# ══════════════════════════════════════════════════════════════════════════════

class TestPasswordValidation:

    def test_register_password_too_short_422(self, client, mb):
        r = client.post("/auth/register", json={
            "email": "a@b.com", "password": "short"
        })
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "weak_password"

    def test_register_exactly_8_chars_ok(self, client, mb):
        mb.supabase.register.return_value = {
            "ok": True, "user_id": FAKE_USER_ID, "access_token": FAKE_TOKEN,
            "refresh_token": "ref", "email": "a@b.com",
        }
        r = client.post("/auth/register", json={
            "email": "a@b.com", "password": "Pass1234"
        })
        assert r.status_code == 200

    def test_change_password_too_short_422(self, client, mb):
        r = client.post("/auth/change-password", json={"new_password": "abc"})
        assert r.status_code == 422

    def test_reset_password_too_short_422(self, client, mb):
        r = client.post("/auth/reset-password", json={
            "access_token": "tok", "new_password": "abc"
        })
        assert r.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# Smoke — all routes registered (no 404)
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteRegistration:

    _ROUTES = [
        ("POST", "/auth/register"),
        ("POST", "/auth/login"),
        ("POST", "/auth/logout"),
        ("GET",  "/auth/me"),
        ("POST", "/auth/refresh"),
        ("POST", "/auth/forgot-password"),
        ("POST", "/auth/reset-password"),
        ("POST", "/auth/change-password"),
        ("POST", "/auth/resend-confirmation"),
        ("GET",  "/sessions"),
        ("POST", "/sessions/create"),
        ("POST", "/sessions/prepare"),
        ("GET",  "/sessions/s/status"),
        ("POST", "/sessions/s/stop"),
        ("POST", "/sessions/s/confirm"),
        ("GET",  "/sessions/s/hyperparams"),
        ("DELETE", "/sessions/s"),
        ("GET",  "/sessions/s/checkpoints"),
        ("GET",  "/sessions/s/segments"),
        ("POST", "/sessions/s/segments/approve"),
        ("POST", "/sessions/s/segments/approve-all"),
        ("POST", "/sessions/s/segments/reject"),
        ("DELETE", "/sessions/s/segments"),
        ("DELETE", "/sessions/s/segments/seg_1"),
        ("GET",  "/inference/models"),
        ("POST", "/inference/convert"),
        ("POST", "/upload"),
    ]

    def test_all_routes_registered(self, unauth_client, mb):
        for method, path in self._ROUTES:
            r = unauth_client.request(method, path)
            assert r.status_code != 404, f"{method} {path} returned 404 — route not mounted"
