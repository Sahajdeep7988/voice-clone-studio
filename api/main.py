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

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.backend import AppBackend

app = FastAPI(title="Voice Clone Studio API", version="1.0.0")

# One backend instance per process — sessions survive across requests.
_backend = AppBackend(base_dir=_ROOT)


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
        ),
    )

    if not result.get("ok"):
        yield _sse("error", result)
        return

    session_id: str = result["session_id"]
    yield _sse("session_created", {"session_id": session_id})

    terminal_states = {"done", "error"}
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
            event_name = "complete" if status == "done" else "error"
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
async def create_session(body: CreateSessionRequest):
    """
    Start a new training pipeline.
    Returns an SSE stream with events:
      - session_created  {"session_id": "..."}
      - progress         {"status", "current_epoch", "total_epochs", "latest_loss", "engine"}
      - complete         {"session_id", "status", "checkpoint_path"}
      - error            {"ok": false, "error": "..."}
    """
    return StreamingResponse(
        _stream_pipeline(
            files=body.files,
            model_name=body.model_name,
            hyperparams_override=body.hyperparams_override,
            max_workers=body.max_workers,
            resume=body.resume,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions")
def list_sessions(user_id: Optional[str] = None):
    """List all sessions, optionally filtered by user_id."""
    return _backend.list_sessions(user_id=user_id)


@app.get("/sessions/{session_id}/status")
def session_status(session_id: str):
    """Get status and engine details for a session."""
    result = _backend.get_session_status(session_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return result


@app.post("/sessions/{session_id}/stop")
def stop_session(session_id: str, body: StopSessionRequest = StopSessionRequest()):
    """Stop or pause training for a session."""
    result = _backend.stop_training(session_id, force=body.force)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return result


@app.post("/inference/convert")
def inference_convert(body: ConvertRequest):
    """Convert audio using a trained model."""
    result = _backend.convert_audio(
        model_path=body.model_path,
        input_audio_path=body.input_audio_path,
        output_path=body.output_path,
        pitch_shift=body.pitch_shift,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Conversion failed"))
    return result
