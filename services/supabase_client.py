"""
Supabase Client
JWT auth + training session sync + token refresh.
Token stored in memory only — never written to disk.
All network calls fail gracefully when offline.
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


class SupabaseClient:

    def __init__(self):
        self._token:         str | None = None
        self._refresh_token: str | None = None
        self._user_id:       str | None = None
        self._client        = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> bool:
        try:
            client   = self._get_client()
            response = client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            if response.session:
                self._store_session(response)
                print(f"[Supabase] Logged in as {email}")
                return True
            print("[Supabase] Login failed: no session returned.")
            return False
        except Exception as e:
            print(f"[Supabase] Login failed (offline or error): {e}")
            return False

    def register(self, email: str, password: str) -> bool:
        try:
            client   = self._get_client()
            response = client.auth.sign_up(
                {"email": email, "password": password}
            )
            if response.user:
                print(f"[Supabase] Registered: {email}")
                if response.session:
                    self._store_session(response)
                return True
            print("[Supabase] Registration failed.")
            return False
        except Exception as e:
            print(f"[Supabase] Registration failed: {e}")
            return False

    def logout(self) -> bool:
        try:
            if self._client and self._token:
                self._client.auth.sign_out()
        except Exception:
            pass
        self._token         = None
        self._refresh_token = None
        self._user_id       = None
        print("[Supabase] Logged out.")
        return True

    def refresh_token(self) -> bool:
        """Refresh the JWT using the stored refresh token."""
        if not self._refresh_token:
            return False
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

    def sync_training_session(self, metadata: dict) -> bool:
        try:
            client = self._get_client()
            record = {
                "user_id":          metadata.get("user_id", self._user_id),
                "model_name":       metadata.get("model_name", ""),
                "status":           metadata.get("status", "completed"),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "epochs_completed": metadata.get("epochs_completed", 0),
                "hardware_profile": metadata.get("hardware_profile", {}),
                "session_id":       metadata.get("session_id"),
                "checkpoint_path":  metadata.get("checkpoint_path"),
                "created_at":       metadata.get(
                    "created_at",
                    datetime.now(timezone.utc).isoformat(),
                ),
            }
            response = (
                client.table("training_sessions")
                .upsert(record, on_conflict="user_id,model_name,created_at")
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

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def access_token(self) -> str | None:
        """The current in-memory JWT access token."""
        return self._token

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None

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

    def _store_session(self, response) -> None:
        self._token         = response.session.access_token
        self._refresh_token = response.session.refresh_token
        self._user_id       = response.user.id

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
