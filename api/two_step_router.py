"""
Two-step pipeline routes (prepare → confirm).
Mounted onto the main app via api/app.py.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.prepare_engine import PrepareEngine
from api.main import get_current_user, _ensure_safe_path

router = APIRouter(tags=["two-step-pipeline"])

# Lazy singleton — shares the same AppBackend instance as api/main.py.
_engine: PrepareEngine | None = None


def _get_engine() -> PrepareEngine:
    global _engine
    if _engine is None:
        from api.main import _backend  # noqa: PLC0415
        _engine = PrepareEngine(_backend)
    return _engine


# ── Request models ────────────────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    files: list[str]
    model_name: str
    max_workers: int = 4
    resume: Optional[bool] = None


class ConfirmRequest(BaseModel):
    hyperparams_override: Optional[dict] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/sessions/prepare", summary="Step 1 — preprocess + generate hyperparams")
def prepare_session(body: PrepareRequest, current=Depends(get_current_user)):
    """
    Run preprocessing and LLM hyperparameter generation for a new model.
    Does NOT start training.
    Returns session_id and generated hyperparams for user review.
    Session status will be 'awaiting_confirmation' on success.
    """
    # FastAPI runs sync route functions in a thread pool automatically.
    safe_files = [_ensure_safe_path(p) for p in body.files]
    result = _get_engine().prepare(
        files=safe_files,
        model_name=body.model_name,
        max_workers=body.max_workers,
        resume=body.resume,
        user_id=current["id"],
    )
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@router.get(
    "/sessions/{session_id}/hyperparams",
    summary="Fetch pending hyperparams for a prepared session",
)
def get_hyperparams(session_id: str, current=Depends(get_current_user)):
    """
    Returns the LLM-generated hyperparams stored on the session.
    Useful for pre-populating a review/edit UI before calling /confirm.
    """
    from api.main import _backend  # noqa: PLC0415

    session = _backend.sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") not in (None, current["id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    hyperparams = session.get("hyperparams")
    if not hyperparams:
        raise HTTPException(
            status_code=404,
            detail="No hyperparams available yet — run /sessions/prepare first",
        )

    return {
        "ok":         True,
        "session_id": session_id,
        "status":     session.get("status"),
        "hyperparams": hyperparams,
    }


@router.post(
    "/sessions/{session_id}/confirm",
    summary="Step 2 — start training with (optionally edited) hyperparams",
)
async def confirm_session(
    session_id: str,
    body: ConfirmRequest = ConfirmRequest(),
    current=Depends(get_current_user),
):
    """
    Start actual training for a session that is in 'awaiting_confirmation' status.
    Optionally pass hyperparams_override to replace any LLM-generated values.
    Returns an SSE stream:
      - training_started
      - progress  (emitted each time the epoch counter advances)
      - complete / error
    """
    return StreamingResponse(
        _get_engine().confirm_and_stream(
            session_id=session_id,
            hyperparams_override=body.hyperparams_override,
            user_id=current["id"],
            access_token=current["token"],
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
