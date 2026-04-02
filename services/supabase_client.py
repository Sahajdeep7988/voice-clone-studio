"""
Supabase Client
JWT auth + training session sync.
All network calls are wrapped to fail gracefully when offline.
Credentials loaded from .env — never written to disk.
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


class SupabaseClient:

    def __init__(self):
        self._token: str | None = None
        self._user_id: str | None = None
        self._client = None  # lazy-loaded supabase client

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> bool:
        """
        Authenticate with Supabase using email/password.
        Token is stored in memory only — never written to disk.
        Returns True on success, False if offline or credentials invalid.
        """
        try:
            client = self._get_client()
            response = client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            if response.session:
                self._token = response.session.access_token
                self._user_id = response.user.id
                print(f"[Supabase] Logged in as {email}")
                return True
            print("[Supabase] Login failed: no session returned.")
            return False
        except Exception as e:
            print(f"[Supabase] Login failed (offline or error): {e}")
            return False

    def register(self, email: str, password: str) -> bool:
        """
        Register a new user.
        Returns True on success, False otherwise.
        """
        try:
            client = self._get_client()
            response = client.auth.sign_up(
                {"email": email, "password": password}
            )
            if response.user:
                print(f"[Supabase] Registered user: {email}")
                # Auto-login after registration if session is present
                if response.session:
                    self._token = response.session.access_token
                    self._user_id = response.user.id
                return True
            print("[Supabase] Registration failed.")
            return False
        except Exception as e:
            print(f"[Supabase] Registration failed (offline or error): {e}")
            return False

    # ------------------------------------------------------------------
    # Training session sync
    # ------------------------------------------------------------------

    def sync_training_session(self, metadata: dict) -> bool:
        """
        Upsert a training session record to the training_sessions table.
        Returns True on success, False if offline.

        Expected metadata keys:
            user_id, model_name, status, duration_seconds,
            epochs_completed, hardware_profile, created_at
        """
        try:
            client = self._get_client()
            record = {
                "user_id":          metadata.get("user_id", self._user_id),
                "model_name":       metadata.get("model_name", ""),
                "status":           metadata.get("status", "completed"),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "epochs_completed": metadata.get("epochs_completed", 0),
                "hardware_profile": metadata.get("hardware_profile", {}),
                "created_at":       metadata.get(
                    "created_at",
                    datetime.now(timezone.utc).isoformat(),
                ),
            }

            # Use upsert on (user_id, model_name, created_at) as composite key
            response = (
                client.table("training_sessions")
                .upsert(record, on_conflict="user_id,model_name,created_at")
                .execute()
            )

            if response.data:
                print(
                    f"[Supabase] Session synced: "
                    f"{record['model_name']} ({record['status']})"
                )
                return True

            print("[Supabase] Sync returned no data.")
            return False

        except Exception as e:
            print(f"[Supabase] Sync failed (offline or error): {e}")
            return False

    def get_training_history(self, user_id: str) -> list:
        """
        Fetch all training sessions for a user.
        Returns empty list if offline or user has no sessions.
        """
        try:
            client = self._get_client()
            response = (
                client.table("training_sessions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"[Supabase] Failed to fetch history (offline or error): {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazy-load and return a Supabase client instance."""
        if self._client is not None:
            return self._client

        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env"
            )

        try:
            from supabase import create_client, Client
        except ImportError:
            raise ImportError(
                "supabase-py not installed. Run: pip install supabase"
            )

        self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return self._client

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None
