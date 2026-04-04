"""
File upload endpoint.
Accepts multipart audio/video files, saves them to a per-user upload directory,
and returns the absolute server-side paths for use in POST /sessions/create
and POST /sessions/prepare.
"""

import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.main import _ROOT, get_current_user

router = APIRouter(tags=["upload"])

# Directory under project root where uploads land.
# Included in _SAFE_DIRS in api/main.py so pipeline path validation passes.
_UPLOADS_ROOT = os.path.abspath(os.path.join(_ROOT, "uploads"))

# Mirror of AudioPreprocessor.SUPPORTED_FORMATS — validated here at the boundary.
_SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac",
    ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
}

# 500 MB per file — generous for long audio recordings.
_MAX_FILE_BYTES = 500 * 1024 * 1024


@router.post("/upload", summary="Upload audio/video files for training")
async def upload_files(
    files: list[UploadFile] = File(...),
    current=Depends(get_current_user),
):
    """
    Upload one or more audio/video files to the server.
    Returns a list of absolute server-side paths ready to pass to
    POST /sessions/create or POST /sessions/prepare.

    Accepted formats: mp3, wav, flac, m4a, aac, ogg, opus, wma, mp4, mkv, avi, mov, webm
    Max size per file: 500 MB
    Files are stored under uploads/{user_id}/ with a UUID prefix to avoid collisions.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    user_dir = os.path.join(_UPLOADS_ROOT, current["id"])
    os.makedirs(user_dir, exist_ok=True)

    saved = []
    errors = []

    for upload in files:
        original_name = upload.filename or "upload"

        # Extension check
        ext = os.path.splitext(original_name)[1].lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            errors.append({
                "file": original_name,
                "error": f"Unsupported format '{ext}'. Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
            })
            continue

        # Sanitise filename — keep only the basename, strip path components
        safe_name = os.path.basename(original_name)
        dest_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        dest_path = os.path.join(user_dir, dest_name)

        # Stream to disk, enforcing the size limit
        bytes_written = 0
        try:
            with open(dest_path, "wb") as fh:
                while True:
                    chunk = await upload.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > _MAX_FILE_BYTES:
                        fh.close()
                        os.remove(dest_path)
                        errors.append({
                            "file": original_name,
                            "error": f"File exceeds the 500 MB limit ({bytes_written / 1024 / 1024:.1f} MB received so far).",
                        })
                        dest_path = None
                        break
                    fh.write(chunk)
        except Exception as e:
            if dest_path and os.path.exists(dest_path):
                os.remove(dest_path)
            errors.append({"file": original_name, "error": str(e)})
            continue

        if dest_path and os.path.exists(dest_path):
            saved.append({
                "original_name": original_name,
                "path":          dest_path,
                "size_bytes":    bytes_written,
            })

    if not saved and errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "All uploads failed.", "errors": errors},
        )

    return JSONResponse(
        status_code=207 if errors else 200,
        content={
            "ok":     True,
            "saved":  saved,
            "paths":  [s["path"] for s in saved],   # ready to pass to /sessions/create
            "errors": errors,
        },
    )
