"""
Supabase Client
JWT auth + training session sync + token refresh.
Token stored in memory only — never written to disk.
All network calls fail gracefully when offline.
"""

import os
import requests
from dotenv import load_dotenv

try:
    from supabase_auth.errors import AuthApiError
except ImportError:
    AuthApiError = Exception  # fallback: still caught, code/status unavailable

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


class SupabaseClient:

    def __init__(self):
        self._client = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> dict:
        try:
            client   = self._get_client()
            response = client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            if response.session:
                print(f"[Supabase] Logged in as {email}")
                return {
                    "ok":            True,
                    "user_id":       response.user.id if response.user else None,
                    "email":         response.user.email if response.user else None,
                    "access_token":  response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                }
            print("[Supabase] Login failed: no session returned.")
            return {"ok": False, "code": "unknown", "detail": "No session returned."}
        except AuthApiError as e:
            print(f"[Supabase] Login error [{e.code}]: {e.message}")
            return {"ok": False, "code": str(e.code), "detail": e.message, "status": e.status}
        except Exception as e:
            print(f"[Supabase] Login failed (network/config): {e}")
            return {"ok": False, "code": "network_error", "detail": str(e)}

    def register(self, email: str, password: str) -> dict:
        try:
            client   = self._get_client()
            response = client.auth.sign_up(
                {"email": email, "password": password}
            )
            if response.user:
                print(f"[Supabase] Registered: {email}")
                return {
                    "ok":            True,
                    "user_id":       response.user.id,
                    "email":         response.user.email,
                    "access_token":  response.session.access_token  if response.session else None,
                    "refresh_token": response.session.refresh_token if response.session else None,
                }
            print("[Supabase] Registration failed: no user returned.")
            return {"ok": False, "code": "unknown", "detail": "Registration failed."}
        except AuthApiError as e:
            print(f"[Supabase] Register error [{e.code}]: {e.message}")
            return {"ok": False, "code": str(e.code), "detail": e.message, "status": e.status}
        except Exception as e:
            print(f"[Supabase] Registration failed (network/config): {e}")
            return {"ok": False, "code": "network_error", "detail": str(e)}

    def logout(self) -> bool:
        try:
            if self._client:
                self._client.auth.sign_out()
        except Exception:
            pass
        print("[Supabase] Logged out.")
        return True

    def refresh_token(self, refresh_token: str | None) -> dict:
        """Refresh a JWT using the provided refresh token."""
        if not refresh_token:
            return {"ok": False}
        try:
            client   = self._get_client()
            response = client.auth.refresh_session(refresh_token)
            if response.session:
                return {
                    "ok": True,
                    "user_id": response.user.id if response.user else None,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                }
        except Exception as e:
            print(f"[Supabase] Token refresh failed: {e}")
        return {"ok": False}

    def forgot_password(self, email: str) -> dict:
        """Send a password-reset email via Supabase GoTrue."""
        try:
            client = self._get_client()
            client.auth.reset_password_for_email(email)
            return {"ok": True}
        except Exception as e:
            print(f"[Supabase] forgot_password failed: {e}")
            return {"ok": False, "error": str(e)}

    def reset_password(self, access_token: str, new_password: str) -> dict:
        """
        Update password using the recovery access_token from the reset-link.
        Calls GoTrue PUT /user directly — stateless, does not touch the singleton session.
        """
        return self._auth_put_user(access_token, {"password": new_password})

    def change_password(self, access_token: str, new_password: str) -> dict:
        """
        Update password for a logged-in user using their current JWT.
        Calls GoTrue PUT /user directly — stateless, does not touch the singleton session.
        """
        return self._auth_put_user(access_token, {"password": new_password})

    def resend_confirmation(self, email: str) -> dict:
        """Resend the email-confirmation link for an unconfirmed account."""
        try:
            client = self._get_client()
            client.auth.resend({"type": "signup", "email": email})
            return {"ok": True}
        except Exception as e:
            print(f"[Supabase] resend_confirmation failed: {e}")
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Training session sync
    # ------------------------------------------------------------------

    # Columns this method is allowed to write. created_at is intentionally
    # excluded — DB DEFAULT handles the first INSERT and the fn_set_updated_at()
    # trigger maintains updated_at on every subsequent write.
    _SYNC_COLUMNS = {
        "user_id", "session_id", "model_name", "status",
        "files", "hyperparams",
        "current_epoch", "total_epochs", "latest_loss",
        "epochs_completed", "duration_seconds", "hardware_profile",
        "checkpoint_path", "segment_manifest", "error_message",
    }

    def sync_training_session(self, metadata: dict, access_token: str | None = None) -> bool:
        """
        Upsert any subset of training_sessions columns.
        session_id is required (upsert conflict target).
        Pass only the keys that changed — unchanged columns are left untouched by Postgres.
        """
        if not metadata.get("session_id"):
            print("[Supabase] sync_training_session: session_id is required")
            return False
        try:
            client = self._get_client()
            if access_token:
                client.postgrest.auth(access_token)

            record = {k: v for k, v in metadata.items() if k in self._SYNC_COLUMNS}

            response = (
                client.table("training_sessions")
                .upsert(record, on_conflict="session_id")
                .execute()
            )
            if response.data:
                print(f"[Supabase] Synced: {record.get('model_name', '?')} ({record.get('status', '?')})")
                return True
            print("[Supabase] Sync returned no data.")
            return False
        except Exception as e:
            print(f"[Supabase] Sync failed (offline or error): {e}")
            return False

    def get_training_history(self, user_id: str) -> list:
        try:
            client   = self._get_client()
            response = (
                client.table("training_sessions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"[Supabase] Failed to fetch history: {e}")
            return []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def get_user_by_token(self, token: str) -> dict | None:
        """
        Validate a JWT and return the user's public info.
        Used by GET /auth/me to authenticate stateless API requests.
        Returns None if the token is invalid or expired.
        """
        try:
            client   = self._get_client()
            response = client.auth.get_user(token)
            if response and response.user:
                u = response.user
                return {
                    "id":         u.id,
                    "email":      u.email,
                    "role":       u.role or "authenticated",
                    "created_at": str(u.created_at),
                }
            return None
        except Exception as e:
            print(f"[Supabase] get_user_by_token failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_put_user(self, access_token: str, attrs: dict) -> dict:
        """
        Stateless GoTrue PUT /user call authenticated with the provided JWT.
        Used for password reset and change-password flows — never mutates
        the singleton client's internal session.
        """
        try:
            url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/user"
            resp = requests.put(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "apikey": SUPABASE_KEY,
                    "Content-Type": "application/json",
                },
                json=attrs,
                timeout=15,
            )
            if resp.status_code == 200:
                return {"ok": True}
            body = resp.json() if resp.content else {}
            return {"ok": False, "error": body.get("message") or body.get("msg") or "Update failed"}
        except Exception as e:
            print(f"[Supabase] _auth_put_user failed: {e}")
            return {"ok": False, "error": str(e)}

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError("supabase-py not installed. Run: pip install supabase")
        self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return self._client
