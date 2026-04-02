"""
Voice Clone Studio — FastAPI Application
Wraps AppBackend; does not modify any existing files.
"""

import asyncio
import json
import sys
import os
from typing import Optional

# Ensure project root is on the path regardless of working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.backend import AppBackend

app = FastAPI(title="Voice Clone Studio API", version="1.0.0")

# One backend instance per process — sessions survive across requests.
_backend = AppBackend(base_dir=_ROOT)


# ---------------------------------------------------------------------------
# Auth + Path Safety
# ---------------------------------------------------------------------------

_SAFE_DIRS = [
    os.path.abspath(os.path.join(_ROOT, "sessions_data")),
    os.path.abspath(os.path.join(_ROOT, "test_inputs")),
    os.path.expanduser("~/voice-clone-studio"),
    os.path.expanduser("~/Desktop/voice-clone-studio"),
    os.path.expanduser("~/VoiceClone"),
    os.path.expanduser("~/.voice_clone_studio"),
    "/tmp",
]


def _ensure_safe_path(path: str, allow_missing: bool = False) -> str:
    abs_path = os.path.abspath(path)
    if not allow_missing and not os.path.exists(abs_path):
        raise HTTPException(status_code=400, detail=f"Path not found: {path}")
    for base in _SAFE_DIRS:
        base_abs = os.path.abspath(base)
        try:
            if os.path.commonpath([abs_path, base_abs]) == base_abs:
                return abs_path
        except ValueError:
            continue
    raise HTTPException(status_code=403, detail=f"Path not allowed: {path}")


def _get_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>",
        )
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token.")
    return token


def get_current_user(token: str = Depends(_get_token)) -> dict:
    user = _backend.supabase.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"id": user["id"], "token": token}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    files: list[str]
    model_name: str
    hyperparams_override: Optional[dict] = None
    max_workers: int = 4
    resume: Optional[bool] = None


class StopSessionRequest(BaseModel):
    force: bool = False


class ConvertRequest(BaseModel):
    model_path: str
    input_audio_path: str
    output_path: str
    pitch_shift: int = 0


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_pipeline(
    files: list[str],
    model_name: str,
    hyperparams_override: Optional[dict],
    max_workers: int,
    resume: Optional[bool],
    user_id: Optional[str],
    access_token: Optional[str],
):
    """
    Generator that:
    1. Kicks off the pipeline in async_mode=True.
    2. Polls get_session_status() every second, yielding SSE events.
    3. Terminates when status is 'done' or 'error'.
    """
    loop = asyncio.get_event_loop()

    # Start the pipeline off the event loop so the blocking call doesn't
    # freeze the SSE stream.
    result = await loop.run_in_executor(
        None,
        lambda: _backend.run_pipeline(
            files=files,
            model_name=model_name,
            hyperparams_override=hyperparams_override,
            async_mode=True,
            max_workers=max_workers,
            resume=resume,
            user_id=user_id,
            access_token=access_token,
        ),
    )

    if not result.get("ok"):
        yield _sse("error", result)
        return

    session_id: str = result["session_id"]
    yield _sse("session_created", {"session_id": session_id})

    terminal_states = {"done", "error", "paused"}
    last_epoch = -1

    while True:
        await asyncio.sleep(1)

        status_result = await loop.run_in_executor(
            None,
            lambda: _backend.get_session_status(session_id),
        )

        if not status_result.get("ok"):
            yield _sse("error", status_result)
            return

        session = status_result["session"]
        engine  = status_result.get("engine", {})
        status  = session.get("status", "unknown")

        # Emit progress event only when the epoch advances.
        current_epoch = session.get("current_epoch", 0)
        if current_epoch != last_epoch:
            last_epoch = current_epoch
            yield _sse(
                "progress",
                {
                    "session_id":    session_id,
                    "status":        status,
                    "current_epoch": current_epoch,
                    "total_epochs":  session.get("total_epochs", 0),
                    "latest_loss":   session.get("latest_loss"),
                    "engine":        engine,
                },
            )

        if status in terminal_states:
            if status == "done":
                event_name = "complete"
            elif status == "paused":
                event_name = "paused"
            else:
                event_name = "error"
            yield _sse(
                event_name,
                {
                    "session_id":      session_id,
                    "status":          status,
                    "checkpoint_path": session.get("checkpoint_path"),
                    "error_message":   session.get("error_message"),
                },
            )
            return


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/sessions/create")
async def create_session(body: CreateSessionRequest, current=Depends(get_current_user)):
    """
    Start a new training pipeline.
    Returns an SSE stream with events:
      - session_created  {"session_id": "..."}
      - progress         {"status", "current_epoch", "total_epochs", "latest_loss", "engine"}
      - complete         {"session_id", "status", "checkpoint_path"}
      - error            {"ok": false, "error": "..."}
    """
    safe_files = [_ensure_safe_path(p) for p in body.files]
    return StreamingResponse(
        _stream_pipeline(
            files=safe_files,
            model_name=body.model_name,
            hyperparams_override=body.hyperparams_override,
            max_workers=body.max_workers,
            resume=body.resume,
            user_id=current["id"],
            access_token=current["token"],
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions")
def list_sessions(user_id: Optional[str] = None, current=Depends(get_current_user)):
    """List all sessions, optionally filtered by user_id."""
    if user_id and user_id != current["id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return _backend.list_sessions(user_id=current["id"])


@app.get("/sessions/{session_id}/status")
def session_status(session_id: str, current=Depends(get_current_user)):
    """Get status and engine details for a session."""
    result = _backend.get_session_status(session_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    if result.get("session", {}).get("user_id") not in (None, current["id"]):
        raise HTTPException(status_code=403, detail="Forbidden")
    return result


@app.post("/sessions/{session_id}/stop")
def stop_session(
    session_id: str,
    body: StopSessionRequest = StopSessionRequest(),
    current=Depends(get_current_user),
):
    """Stop or pause training for a session."""
    sess = _backend.sessions.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found")
    if sess.get("user_id") not in (None, current["id"]):
        raise HTTPException(status_code=403, detail="Forbidden")
    result = _backend.stop_training(session_id, force=body.force)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return result


@app.post("/inference/convert")
def inference_convert(body: ConvertRequest, current=Depends(get_current_user)):
    """Convert audio using a trained model."""
    model_path = _ensure_safe_path(body.model_path)
    input_audio_path = _ensure_safe_path(body.input_audio_path)
    output_path = _ensure_safe_path(body.output_path, allow_missing=True)
    result = _backend.convert_audio(
        model_path=model_path,
        input_audio_path=input_audio_path,
        output_path=output_path,
        pitch_shift=body.pitch_shift,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Conversion failed"))
    return result
