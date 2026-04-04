"""
Segment curation routes.
Exposes get/approve/delete operations on preprocessed audio segments.
All routes are scoped to the owning user via JWT.
"""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel

from api.main import get_current_user, _backend, _ensure_safe_path

router = APIRouter(tags=["segments"])


# ── Request models ────────────────────────────────────────────────────────────

class ApproveSegmentsRequest(BaseModel):
    approved_paths: list[str]


class DeleteSegmentRequest(BaseModel):
    path: str


class RejectSegmentsRequest(BaseModel):
    paths: list[str]


# ── Ownership guard ───────────────────────────────────────────────────────────

def _get_owned_session(session_id: str, user_id: str) -> dict:
    sess = _backend.sessions.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess.get("user_id") not in (None, user_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    return sess


def _resolve_segment_path_by_id(session_id: str, segment_id: str) -> str:
    """
    Resolve a UI-friendly segment_id to an absolute segment path.
    Accepted IDs:
      - exact filename: seg_0001.wav
      - filename stem:  seg_0001
    """
    result = _backend.get_segments(session_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Session not found"))

    for seg in result.get("segments", []):
        path = seg.get("path")
        if not path:
            continue
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        if segment_id in (name, stem):
            return path

    raise HTTPException(status_code=404, detail="Segment not found")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/segments",
    summary="List all preprocessed segments for a session",
)
def get_segments(session_id: str, current=Depends(get_current_user)):
    """
    Returns the segment manifest for a session.
    Each entry contains: path, duration_s, size_bytes, approved.
    Only available after preprocessing has completed.
    """
    _get_owned_session(session_id, current["id"])
    result = _backend.get_segments(session_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return result


@router.post(
    "/sessions/{session_id}/segments/approve",
    summary="Approve a subset of segments for training",
)
def approve_segments(
    session_id: str,
    body: ApproveSegmentsRequest,
    current=Depends(get_current_user),
):
    """
    Mark specific segments as approved. Only approved segments are used during training.
    Pass the full list of paths that should be approved — any not in the list are unapproved.
    Paths are validated against allowed directories.

    Typically called before POST /sessions/{id}/confirm to curate the training dataset.
    """
    _get_owned_session(session_id, current["id"])

    # Validate each path is within safe directories and exists on disk
    safe_paths = []
    for p in body.approved_paths:
        try:
            safe_paths.append(_ensure_safe_path(p))
        except HTTPException:
            raise HTTPException(
                status_code=400,
                detail=f"Path not allowed or not found: {p}",
            )

    result = _backend.approve_segments(session_id, safe_paths)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Approval failed"))
    return result


@router.post(
    "/sessions/{session_id}/segments/approve-all",
    summary="Approve all segments for a session",
)
def approve_all_segments(session_id: str, current=Depends(get_current_user)):
    """
    Mark every segment in the manifest as approved.
    Shortcut for when no curation is needed.
    """
    _get_owned_session(session_id, current["id"])
    result = _backend.get_segments(session_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    all_paths = [s["path"] for s in result.get("segments", [])]
    approved = _backend.approve_segments(session_id, all_paths)
    if not approved.get("ok"):
        raise HTTPException(status_code=500, detail=approved.get("error", "Failed"))
    return approved


@router.post(
    "/sessions/{session_id}/segments/reject",
    summary="Reject a subset of segments for training",
)
def reject_segments(
    session_id: str,
    body: RejectSegmentsRequest,
    current=Depends(get_current_user),
):
    """
    Mark specific segments as rejected by approving all remaining segments.
    Complements /segments/approve for explicit UI reject actions.
    """
    _get_owned_session(session_id, current["id"])

    result = _backend.get_segments(session_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))

    reject_set: set[str] = set()
    for p in body.paths:
        try:
            reject_set.add(os.path.abspath(_ensure_safe_path(p)))
        except HTTPException:
            raise HTTPException(status_code=400, detail=f"Path not allowed or not found: {p}")

    all_paths = [s["path"] for s in result.get("segments", []) if "path" in s]
    approved_paths = [p for p in all_paths if os.path.abspath(p) not in reject_set]

    approved = _backend.approve_segments(session_id, approved_paths)
    if not approved.get("ok"):
        raise HTTPException(status_code=500, detail=approved.get("error", "Rejection failed"))

    rejected_count = len([p for p in all_paths if os.path.abspath(p) in reject_set])
    return {
        "ok": True,
        "approved_count": approved.get("approved_count", len(approved_paths)),
        "rejected_count": rejected_count,
    }


@router.delete(
    "/sessions/{session_id}/segments",
    summary="Delete a single segment from the manifest and disk",
)
def delete_segment(
    session_id: str,
    path: Optional[str] = Query(None, description="Absolute segment path to delete"),
    body: Optional[DeleteSegmentRequest] = Body(None),
    current=Depends(get_current_user),
):
    """
    Permanently deletes one segment file and removes it from the manifest.
    The path must be within allowed directories.
    Useful for removing low-quality or unwanted clips before training.
    """
    _get_owned_session(session_id, current["id"])

    # Backward compatibility:
    # 1) DELETE .../segments with JSON body {"path": "..."}
    # 2) DELETE .../segments?path=...
    candidate = path or (body.path if body else None)
    if not candidate:
        raise HTTPException(status_code=422, detail="Provide segment path via query '?path=' or request body.")

    try:
        safe_path = _ensure_safe_path(candidate)
    except HTTPException:
        raise HTTPException(status_code=400, detail=f"Path not allowed: {candidate}")

    result = _backend.delete_segment(session_id, safe_path)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail="Segment not found or already deleted")
    return result


@router.delete(
    "/sessions/{session_id}/segments/{segment_id}",
    summary="Delete a segment by id (filename or stem)",
)
def delete_segment_by_id(
    session_id: str,
    segment_id: str,
    current=Depends(get_current_user),
):
    """
    Delete a segment by id to avoid sending absolute paths from UI.
    segment_id accepts either filename (seg_0001.wav) or filename stem (seg_0001).
    """
    _get_owned_session(session_id, current["id"])

    raw_path = _resolve_segment_path_by_id(session_id, segment_id)
    try:
        safe_path = _ensure_safe_path(raw_path)
    except HTTPException:
        raise HTTPException(status_code=400, detail=f"Path not allowed: {raw_path}")

    result = _backend.delete_segment(session_id, safe_path)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail="Segment not found or already deleted")
    return result
