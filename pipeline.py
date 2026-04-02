#!/usr/bin/env python3
"""
Voice Clone Studio — CLI Pipeline Runner
Delegates to AppBackend — the same code path a future UI will use.

Usage:
  python pipeline.py --files audio.mp3 --model-name myvoice
  python pipeline.py --files a.mp3 b.wav --model-name myvoice --resume
  python pipeline.py --files audio.mp3 --model-name myvoice \
                     --email user@example.com --password secret
  python pipeline.py --status <session_id>
  python pipeline.py --checkpoints <session_id>
  python pipeline.py --test-checkpoint <session_id> <checkpoint.pth> <test.wav>
  python pipeline.py --list-sessions
"""

import argparse
import json
import sys


def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Voice Clone Studio — backend pipeline CLI"
    )

    # ── Actions ──────────────────────────────────────────────────────
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status",       metavar="SESSION_ID",
                        help="Show status of a session")
    action.add_argument("--checkpoints",  metavar="SESSION_ID",
                        help="List checkpoints for a session")
    action.add_argument("--list-sessions", action="store_true",
                        help="List all sessions")
    action.add_argument("--test-checkpoint", nargs=3,
                        metavar=("SESSION_ID", "CKPT_PATH", "TEST_AUDIO"),
                        help="Test a specific checkpoint")
    action.add_argument("--segments",     metavar="SESSION_ID",
                        help="List segments for a session")

    # ── Pipeline args ────────────────────────────────────────────────
    parser.add_argument("--files",       nargs="+", metavar="FILE")
    parser.add_argument("--model-name",  metavar="NAME")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--resume-session", metavar="SESSION_ID",
                        help="Resume a specific paused session")
    parser.add_argument("--email",       default=None)
    parser.add_argument("--password",    default=None)

    args = parser.parse_args()

    from services.backend import AppBackend
    backend = AppBackend()

    # Optional auth
    if args.email and args.password:
        result = backend.login(args.email, args.password)
        if result["ok"]:
            print(f"[Auth] Logged in (user_id={result['user_id']})")
        else:
            print("[Auth] Login failed — continuing without Supabase sync.")

    # ── Dispatch ─────────────────────────────────────────────────────

    if args.status:
        result = backend.get_session_status(args.status)
        _print_json(result)
        return

    if args.list_sessions:
        result = backend.list_sessions()
        for s in result["sessions"]:
            print(f"  {s['session_id'][:8]}  {s['model_name']:20s}  "
                  f"{s['status']:15s}  epoch={s['current_epoch']}/{s['total_epochs']}  "
                  f"{s['created_at'][:19]}")
        return

    if args.checkpoints:
        result = backend.list_checkpoints(args.checkpoints)
        _print_json(result)
        return

    if args.test_checkpoint:
        session_id, ckpt_path, test_audio = args.test_checkpoint
        result = backend.test_checkpoint(session_id, ckpt_path, test_audio)
        _print_json(result)
        return

    if args.segments:
        result = backend.get_segments(args.segments)
        segs   = result.get("segments", [])
        print(f"  {'PATH':60s}  {'DUR':6s}  {'APPROVED'}")
        for s in segs:
            print(f"  {s['path']:60s}  {s['duration_s']:5.1f}s  {s.get('approved', True)}")
        return

    if args.resume_session:
        print(f"[Pipeline] Resuming session {args.resume_session}...")
        result = backend.resume_pipeline(args.resume_session, async_mode=False)
        _print_json(result)
        return

    # ── Default: full pipeline run ────────────────────────────────────
    if not args.files or not args.model_name:
        parser.print_help()
        sys.exit(1)

    print(f"\n[Pipeline] Starting: model='{args.model_name}' "
          f"files={len(args.files)} resume={args.resume}")

    result = backend.run_pipeline(
        files=args.files,
        model_name=args.model_name,
        async_mode=False,
    )

    _print_json(result)


def _print_json(obj: dict) -> None:
    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    run_pipeline()
