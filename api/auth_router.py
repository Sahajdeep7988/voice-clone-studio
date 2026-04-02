"""
Auth routes — wraps SupabaseClient via AppBackend.
All heavy network calls are on the SupabaseClient; this file is pure routing.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/auth", tags=["auth"])


def _supabase():
    """Return the SupabaseClient from the shared AppBackend singleton."""
    from api.main import _backend  # noqa: PLC0415
    return _backend.supabase


# ── Request models ────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    email: EmailStr
    password: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register a new user")
def register(body: AuthRequest):
    """
    Create a new Supabase account.
    Supabase sends a confirmation email if email-confirm is enabled in the project.
    Returns user_id once the account is created.
    """
    sb = _supabase()
    ok = sb.register(body.email, body.password)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Registration failed — the email may already be in use or the password is too weak.",
        )
    return {
        "ok":      True,
        "user_id": sb.user_id,
        "message": "Registration successful.",
    }


@router.post("/login", summary="Login with email + password")
def login(body: AuthRequest):
    """
    Authenticate with Supabase and receive a JWT access token.
    Pass the token in subsequent requests as: Authorization: Bearer <token>
    """
    sb = _supabase()
    ok = sb.login(body.email, body.password)
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
        )
    return {
        "ok":      True,
        "user_id": sb.user_id,
        "token":   sb.access_token,
        "message": "Login successful.",
    }


@router.post("/logout", summary="Logout the current user")
def logout():
    """
    Revoke the current Supabase session and clear in-memory credentials.
    After this call the server-side token is no longer valid.
    """
    _supabase().logout()
    return {"ok": True}


@router.get("/me", summary="Return the current authenticated user")
def me(authorization: Optional[str] = Header(None)):
    """
    Validate the Bearer token from the Authorization header and return user info.
    Expects:  Authorization: Bearer <jwt>
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>",
        )
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token.")

    user = _supabase().get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    return {"ok": True, "user": user}
