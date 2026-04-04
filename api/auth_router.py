"""
Auth routes — wraps SupabaseClient via AppBackend.
All heavy network calls are on the SupabaseClient; this file is pure routing.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/auth", tags=["auth"])

MIN_PASSWORD_LEN = 8


def _supabase():
    from api.main import _backend  # noqa: PLC0415
    return _backend.supabase


def _cache():
    from api.main import _token_cache  # noqa: PLC0415
    return _token_cache


def _prime_cache(access_token: str, user_id: str) -> None:
    """Pre-warm the token cache after a successful login or register."""
    _cache().set(access_token, {"id": user_id, "token": access_token})


# ── Request models ────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    # access_token is the recovery JWT from the reset-link URL fragment
    # (#access_token=...&type=recovery)
    access_token: str
    new_password: str


class ChangePasswordRequest(BaseModel):
    new_password: str


class ResendConfirmationRequest(BaseModel):
    email: EmailStr


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LEN:
        raise HTTPException(
            status_code=422,
            detail={
                "code":    "weak_password",
                "message": f"Password must be at least {MIN_PASSWORD_LEN} characters.",
            },
        )


def _http_from_auth_error(result: dict) -> HTTPException:
    """
    Convert a SupabaseClient error dict to the right HTTPException.
    GoTrue error codes: https://supabase.com/docs/reference/self-hosting-auth/error-codes
    """
    code   = result.get("code", "")
    detail = result.get("detail", "")

    # ── Login-specific ────────────────────────────────────────────────────────
    if code == "email_not_confirmed":
        return HTTPException(
            status_code=403,
            detail={
                "code":    "email_not_confirmed",
                "message": "Please confirm your email before logging in. "
                           "Check your inbox or use POST /auth/resend-confirmation.",
            },
        )
    if code == "invalid_credentials":
        return HTTPException(
            status_code=401,
            detail={"code": "invalid_credentials", "message": "Invalid email or password."},
        )
    if code == "user_banned":
        return HTTPException(
            status_code=403,
            detail={"code": "user_banned", "message": "This account has been suspended."},
        )

    # ── Register-specific ─────────────────────────────────────────────────────
    if code in ("email_exists", "user_already_exists"):
        return HTTPException(
            status_code=409,
            detail={
                "code":    "email_in_use",
                "message": "An account with this email already exists.",
            },
        )
    if code == "weak_password":
        return HTTPException(
            status_code=422,
            detail={"code": "weak_password", "message": detail or "Password is too weak."},
        )
    if code == "signup_disabled":
        return HTTPException(
            status_code=403,
            detail={"code": "signup_disabled", "message": "New registrations are currently disabled."},
        )
    if code == "email_address_invalid":
        return HTTPException(
            status_code=422,
            detail={"code": "invalid_email", "message": "The email address is invalid."},
        )

    # ── Shared ────────────────────────────────────────────────────────────────
    if code in ("over_request_rate_limit", "over_email_send_rate_limit"):
        return HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": "Too many requests. Please wait before trying again."},
        )
    if code == "network_error":
        return HTTPException(
            status_code=503,
            detail={"code": "service_unavailable", "message": "Authentication service unavailable. Try again later."},
        )
    if code == "otp_expired":
        return HTTPException(
            status_code=400,
            detail={"code": "token_expired", "message": "The reset link has expired. Request a new one."},
        )
    if code == "same_password":
        return HTTPException(
            status_code=422,
            detail={"code": "same_password", "message": "New password must be different from the current password."},
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    status = result.get("status", 400)
    if not isinstance(status, int) or status < 400:
        status = 400
    return HTTPException(
        status_code=status,
        detail={"code": code or "auth_error", "message": detail or "Authentication error."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register a new user")
def register(body: AuthRequest):
    """
    Create a new Supabase account.

    Returns `token` immediately when Supabase email-confirm is OFF.
    When email-confirm is ON, `token` is null and `needs_confirmation` is true
    — the UI should show "check your email" instead of attempting to log in.
    """
    _validate_password(body.password)

    result = _supabase().register(body.email, body.password)
    if not result.get("ok"):
        raise _http_from_auth_error(result)

    needs_confirmation = result.get("access_token") is None
    if not needs_confirmation:
        _prime_cache(result["access_token"], result["user_id"])

    return {
        "ok":                True,
        "user_id":           result["user_id"],
        "email":             result.get("email"),
        "token":             result.get("access_token"),
        "refresh_token":     result.get("refresh_token"),
        "needs_confirmation": needs_confirmation,
        "message": (
            "Registration successful. Check your email to confirm your account."
            if needs_confirmation
            else "Registration successful."
        ),
    }


@router.post("/login", summary="Login with email + password")
def login(body: AuthRequest):
    """
    Authenticate and receive a JWT pair.
    Store `refresh_token` and call POST /auth/refresh before the JWT expires (1 h default).
    The returned `token` is pre-warmed in the server cache — no extra round-trip on first use.
    """
    result = _supabase().login(body.email, body.password)
    if not result.get("ok"):
        raise _http_from_auth_error(result)

    # Pre-warm cache so the first protected request hits no Supabase call.
    _prime_cache(result["access_token"], result["user_id"])

    return {
        "ok":            True,
        "user_id":       result["user_id"],
        "email":         result.get("email"),
        "token":         result["access_token"],
        "refresh_token": result.get("refresh_token"),
        "message":       "Login successful.",
    }


@router.post("/refresh", summary="Refresh an expired JWT")
def refresh(body: RefreshRequest):
    """
    Exchange a refresh_token for a new access_token + refresh_token pair.
    The old refresh_token is invalidated server-side — always persist the new one.
    """
    result = _supabase().refresh_token(body.refresh_token)
    if not result.get("ok"):
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_refresh_token", "message": "Invalid or expired refresh token."},
        )

    _prime_cache(result["access_token"], result["user_id"])

    return {
        "ok":            True,
        "user_id":       result["user_id"],
        "token":         result["access_token"],
        "refresh_token": result["refresh_token"],
    }


@router.post("/forgot-password", summary="Request a password reset email")
def forgot_password(body: ForgotPasswordRequest):
    """
    Sends a password reset link to the given address.
    Always returns 200 — never reveals whether the email is registered.

    The link URL fragment contains: #access_token=...&type=recovery
    Extract that token and pass it to POST /auth/reset-password.
    """
    result = _supabase().forgot_password(body.email)
    # Log failures server-side without exposing them to the caller.
    if not result.get("ok"):
        code = result.get("code", "")
        if code in ("over_request_rate_limit", "over_email_send_rate_limit"):
            raise HTTPException(
                status_code=429,
                detail={"code": "rate_limited", "message": "Too many requests. Please wait before trying again."},
            )
        print(f"[Auth] forgot_password silenced error: {result}")
    return {"ok": True, "message": "If that email is registered, a reset link has been sent."}


@router.post("/reset-password", summary="Set a new password via recovery token")
def reset_password(body: ResetPasswordRequest):
    """
    Complete the password-reset flow.

    `access_token` is the recovery JWT from the reset-link URL fragment.
    It is single-use and expires after a short window (typically 1 hour).
    After success the token is evicted from cache — the user must log in again.
    """
    if not body.access_token.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "missing_token", "message": "access_token is required."},
        )
    _validate_password(body.new_password)

    result = _supabase().reset_password(body.access_token, body.new_password)
    if not result.get("ok"):
        raise _http_from_auth_error(result)

    # Evict in case the recovery token was somehow cached.
    _cache().invalidate(body.access_token)
    return {"ok": True, "message": "Password updated. Please log in with your new password."}


@router.post("/change-password", summary="Change password for a logged-in user")
def change_password(
    body: ChangePasswordRequest,
    current=Depends(lambda: None),  # placeholder — see below
    authorization: Optional[str] = Header(None),
):
    """
    Update the password for the currently authenticated user.
    Requires: Authorization: Bearer <token>

    Uses the token cache — no extra Supabase call to validate identity.
    After success the token is evicted; re-login is required.
    """
    # Resolve token manually (same logic as _get_token / get_current_user)
    # so we can both authenticate and then immediately invalidate on success.
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "missing_token", "message": "Missing or malformed Authorization header."},
        )
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail={"code": "missing_token", "message": "Empty token."})

    _validate_password(body.new_password)

    # Validate via cache first, then Supabase on miss.
    cached = _cache().get(token)
    if not cached:
        user = _supabase().get_user_by_token(token)
        if not user:
            raise HTTPException(status_code=401, detail={"code": "invalid_token", "message": "Invalid or expired token."})

    result = _supabase().change_password(token, body.new_password)
    if not result.get("ok"):
        raise _http_from_auth_error(result)

    _cache().invalidate(token)
    return {"ok": True, "message": "Password updated. Please log in with your new password."}


@router.post("/resend-confirmation", summary="Resend email confirmation link")
def resend_confirmation(body: ResendConfirmationRequest):
    """
    Resend the signup confirmation email for an unconfirmed account.
    Always returns 200 — never reveals whether the email is registered.
    """
    result = _supabase().resend_confirmation(body.email)
    if not result.get("ok"):
        code = result.get("code", "")
        if code in ("over_request_rate_limit", "over_email_send_rate_limit"):
            raise HTTPException(
                status_code=429,
                detail={"code": "rate_limited", "message": "Too many requests. Please wait before trying again."},
            )
        print(f"[Auth] resend_confirmation silenced error: {result}")
    return {"ok": True, "message": "If that account exists and is unconfirmed, a new link has been sent."}


@router.post("/logout", summary="Logout the current user")
def logout(authorization: Optional[str] = Header(None)):
    """
    Revoke the Supabase session and immediately evict the token from cache.
    The client must also discard its stored tokens.
    """
    _supabase().logout()
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        if token:
            _cache().invalidate(token)
    return {"ok": True}


@router.get("/me", summary="Return the current authenticated user")
def me(authorization: Optional[str] = Header(None)):
    """
    Validate the Bearer token and return the user's profile.
    Served from cache within the 30-second TTL — no Supabase round-trip on hot paths.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "missing_token", "message": "Missing or malformed Authorization header."},
        )
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail={"code": "missing_token", "message": "Empty token."})

    # Cache hit — only use it if we have the full profile (email, role, etc.).
    # A partial hit (only id+token, from get_current_user or _prime_cache) falls
    # through to Supabase so we always return a consistent user dict.
    cached = _cache().get(token)
    if cached and "email" in cached:
        user_data = {k: v for k, v in cached.items() if k != "token"}
        return {"ok": True, "user": user_data, "cached": True}

    user = _supabase().get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail={"code": "invalid_token", "message": "Invalid or expired token."})

    # Cache the full profile so subsequent hits are consistent.
    _cache().set(token, {**user, "token": token})
    return {"ok": True, "user": user, "cached": False}
