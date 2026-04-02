"""
Supabase Client
JWT auth + training session sync + token refresh.
Token stored in memory only — never written to disk.
All network calls fail gracefully when offline.
"""

import os
from dotenv import load_dotenv

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
                    "ok": True,
                    "user_id": response.user.id if response.user else None,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                }
            print("[Supabase] Login failed: no session returned.")
            return {"ok": False}
        except Exception as e:
            print(f"[Supabase] Login failed (offline or error): {e}")
            return {"ok": False}

    def register(self, email: str, password: str) -> dict:
        try:
            client   = self._get_client()
            response = client.auth.sign_up(
                {"email": email, "password": password}
            )
            if response.user:
                print(f"[Supabase] Registered: {email}")
                return {
                    "ok": True,
                    "user_id": response.user.id,
                    "access_token": response.session.access_token if response.session else None,
                    "refresh_token": response.session.refresh_token if response.session else None,
                }
            print("[Supabase] Registration failed.")
            return {"ok": False}
        except Exception as e:
            print(f"[Supabase] Registration failed: {e}")
            return {"ok": False}

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
        try:
            client   = self._get_client()
            response = client.auth.refresh_session(self._refresh_token)
            if response.session:
                self._store_session(response)
                print("[Supabase] Token refreshed.")
                return True
        except Exception as e:
            print(f"[Supabase] Token refresh failed: {e}")
        return False

    # ------------------------------------------------------------------
    # Training session sync
    # ------------------------------------------------------------------

    def sync_training_session(self, metadata: dict, access_token: str | None = None) -> bool:
        try:
            client = self._get_client()
            if access_token:
                client.postgrest.auth(access_token)

            # Bug-fixes vs original:
            #   1. on_conflict now targets session_id (the actual UNIQUE column).
            #      The old target (user_id,model_name,created_at) had no UNIQUE
            #      constraint, so every call was a blind INSERT that silently failed.
            #   2. created_at is excluded from the upsert record so the DB default
            #      (NOW()) is used on INSERT and the value is never overwritten on
            #      subsequent UPDATE syncs.  Sending it would reset the creation
            #      timestamp to the current time on every training-complete event.
            record = {
                "user_id":          metadata.get("user_id"),
                "model_name":       metadata.get("model_name", ""),
                "status":           metadata.get("status", "completed"),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "epochs_completed": metadata.get("epochs_completed", 0),
                "hardware_profile": metadata.get("hardware_profile", {}),
                "session_id":       metadata.get("session_id"),
                "checkpoint_path":  metadata.get("checkpoint_path"),
                # created_at intentionally omitted — DB DEFAULT handles first INSERT;
                # updated_at is auto-maintained by the fn_set_updated_at() trigger.
            }
            response = (
                client.table("training_sessions")
                .upsert(record, on_conflict="session_id")
                .execute()
            )
            if response.data:
                print(f"[Supabase] Synced: {record['model_name']} ({record['status']})")
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
