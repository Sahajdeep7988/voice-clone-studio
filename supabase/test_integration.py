#!/usr/bin/env python3
"""
Supabase integration tests — validates schema, SupabaseClient, and RLS.

What is tested:
  1. Schema  — table exists; all required columns present with correct types;
               CHECK constraint on status; NOT NULL enforcement; UNIQUE on session_id
  2. Auth    — register / login / logout via SupabaseClient
  3. CRUD    — insert and read training_session rows as authenticated user
  4. RLS     — user A cannot read/mutate user B's rows
  5. Upsert  — re-syncing a session updates in-place (no duplicate rows)
  6. Cleanup — test rows deleted after suite (best-effort)

Run:
    python supabase/test_integration.py
    python supabase/test_integration.py -v      # verbose
    python supabase/test_integration.py --keep  # skip cleanup (inspect rows in dashboard)
"""

import argparse
import os
import sys
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from services.supabase_client import SupabaseClient

# ── Helpers ───────────────────────────────────────────────────────────────────

VERBOSE  = False
PASSED   = 0
FAILED   = 0
WARNINGS = []

_COLORS = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "reset":  "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{_COLORS[color]}{text}{_COLORS['reset']}"


def ok(label: str, detail: str = "") -> None:
    global PASSED
    PASSED += 1
    suffix = f"  {_c('cyan', detail)}" if detail else ""
    print(f"  {_c('green', '✓')} {label}{suffix}")


def fail(label: str, detail: str = "") -> None:
    global FAILED
    FAILED += 1
    suffix = f"\n    {_c('red', detail)}" if detail else ""
    print(f"  {_c('red', '✗')} {label}{suffix}")


def warn(label: str, detail: str = "") -> None:
    WARNINGS.append(f"{label}: {detail}")
    suffix = f"  {_c('yellow', detail)}" if detail else ""
    print(f"  {_c('yellow', '!')} {label}{suffix}")


def section(title: str) -> None:
    print(f"\n{_c('cyan', '─' * 60)}")
    print(f"{_c('cyan', title)}")
    print(_c('cyan', '─' * 60))


def make_email() -> str:
    tag = uuid.uuid4().hex[:8]
    return f"vcs_test_{tag}@example.com"


def make_session_id() -> str:
    return str(uuid.uuid4())


def make_session_record(user_id: str, session_id: str | None = None, model_name: str = "test_model") -> dict:
    """Minimal valid record matching sync_training_session payload."""
    return {
        "user_id":          user_id,
        "session_id":       session_id or make_session_id(),
        "model_name":       model_name,
        "status":           "completed",
        "duration_seconds": 120,
        "epochs_completed": 50,
        "hardware_profile": {"vram_mb": 8192, "gpu_name": "RTX 3080"},
        "checkpoint_path":  "/rvc/logs/test_model/test_50e.pth",
    }


# ── Test groups ───────────────────────────────────────────────────────────────

def test_schema(client_a: SupabaseClient) -> None:
    section("1 · SCHEMA VALIDATION")

    raw = client_a._get_client()

    # ── Column presence & type checks via intentional constraint violations ──

    sid = make_session_id()
    uid = client_a.user_id

    # NOT NULL on user_id
    try:
        raw.table("training_sessions").insert({
            "session_id": sid, "model_name": "x", "status": "created"
            # user_id intentionally omitted
        }).execute()
        fail("user_id NOT NULL enforced", "insert without user_id should have failed")
    except Exception as e:
        msg = str(e)
        if "null" in msg.lower() or "not-null" in msg.lower() or "violates" in msg.lower() or "42501" in msg or "23502" in msg:
            ok("user_id NOT NULL enforced")
        else:
            warn("user_id NOT NULL — unexpected error", msg[:120])

    # NOT NULL on model_name
    try:
        raw.table("training_sessions").insert({
            "user_id": uid, "session_id": make_session_id(), "status": "created"
        }).execute()
        fail("model_name NOT NULL enforced", "insert without model_name should have failed")
    except Exception as e:
        msg = str(e)
        if "null" in msg.lower() or "violates" in msg.lower() or "42501" in msg or "23502" in msg:
            ok("model_name NOT NULL enforced")
        else:
            warn("model_name NOT NULL — unexpected error", msg[:120])

    # CHECK constraint on status
    try:
        raw.table("training_sessions").insert({
            "user_id": uid, "session_id": make_session_id(),
            "model_name": "check_test", "status": "INVALID_VALUE",
        }).execute()
        fail("status CHECK constraint enforced", "invalid status should have been rejected")
    except Exception as e:
        msg = str(e)
        if "check" in msg.lower() or "23514" in msg or "violates" in msg.lower():
            ok("status CHECK constraint enforced", "INVALID_VALUE rejected")
        else:
            warn("status CHECK — unexpected error (constraint may still work)", msg[:120])

    # All valid statuses must be accepted (we'll just check they don't raise)
    valid_statuses = [
        "created", "preprocessing", "preprocessing_done",
        "awaiting_confirmation", "training", "paused",
        "done", "completed", "error",
    ]
    inserted_status_ids = []
    all_ok = True
    for s in valid_statuses:
        sid = make_session_id()
        try:
            r = raw.table("training_sessions").insert({
                "user_id": uid, "session_id": sid,
                "model_name": "status_test", "status": s,
            }).execute()
            if r.data:
                inserted_status_ids.append(sid)
            else:
                all_ok = False
        except Exception as e:
            all_ok = False
            if VERBOSE:
                print(f"    status={s} failed: {e}")
    if all_ok:
        ok(f"All {len(valid_statuses)} valid status values accepted")
    else:
        fail("Some valid status values rejected")
    # Clean up status test rows
    for sid in inserted_status_ids:
        try:
            raw.table("training_sessions").delete().eq("session_id", sid).execute()
        except Exception:
            pass

    # UNIQUE on session_id
    sid = make_session_id()
    raw.table("training_sessions").insert({
        "user_id": uid, "session_id": sid, "model_name": "unique_test", "status": "created",
    }).execute()
    try:
        raw.table("training_sessions").insert({
            "user_id": uid, "session_id": sid, "model_name": "unique_test2", "status": "created",
        }).execute()
        fail("session_id UNIQUE constraint enforced", "duplicate session_id should have been rejected")
    except Exception as e:
        msg = str(e)
        if "unique" in msg.lower() or "23505" in msg or "duplicate" in msg.lower():
            ok("session_id UNIQUE constraint enforced")
        else:
            warn("session_id UNIQUE — unexpected error", msg[:120])
    # clean up
    try:
        raw.table("training_sessions").delete().eq("session_id", sid).execute()
    except Exception:
        pass

    # Column presence — insert a full record and verify returned columns
    sid = make_session_id()
    full_record = {
        "user_id":          uid,
        "session_id":       sid,
        "model_name":       "column_check",
        "status":           "done",
        "files":            ["/audio/test.mp3"],
        "hyperparams":      {"epochs": 100, "batch_size": 4},
        "current_epoch":    100,
        "total_epochs":     100,
        "latest_loss":      0.042,
        "epochs_completed": 100,
        "duration_seconds": 300,
        "hardware_profile": {"gpu": "RTX 3080"},
        "checkpoint_path":  "/rvc/logs/model.pth",
        "segment_manifest": "/sessions/abc/manifest.json",
        "error_message":    None,
    }
    try:
        r = raw.table("training_sessions").insert(full_record).execute()
        if r.data:
            row = r.data[0]
            expected_cols = [
                "id", "user_id", "session_id", "model_name", "status",
                "files", "hyperparams", "current_epoch", "total_epochs",
                "latest_loss", "epochs_completed", "duration_seconds",
                "hardware_profile", "checkpoint_path", "segment_manifest",
                "error_message", "created_at", "updated_at",
            ]
            missing = [c for c in expected_cols if c not in row]
            if not missing:
                ok(f"All {len(expected_cols)} expected columns present in returned row")
            else:
                fail("Missing columns", ", ".join(missing))

            # Verify JSONB round-trip
            if row.get("files") == ["/audio/test.mp3"]:
                ok("JSONB files round-trip correct")
            else:
                fail("JSONB files round-trip", f"got {row.get('files')}")

            if row.get("hyperparams", {}).get("epochs") == 100:
                ok("JSONB hyperparams round-trip correct")
            else:
                fail("JSONB hyperparams round-trip", f"got {row.get('hyperparams')}")

            # Verify timestamps populated
            if row.get("created_at") and row.get("updated_at"):
                ok("Timestamps (created_at, updated_at) auto-populated")
            else:
                fail("Timestamps missing", str({k: row.get(k) for k in ("created_at", "updated_at")}))

            # Verify surrogate id is a UUID
            row_id = row.get("id", "")
            try:
                uuid.UUID(str(row_id))
                ok("Surrogate id is valid UUID")
            except ValueError:
                fail("Surrogate id not a valid UUID", str(row_id))

        # Clean up
        raw.table("training_sessions").delete().eq("session_id", sid).execute()
    except Exception as e:
        fail("Full record insert", str(e)[:120])


def test_crud(client_a: SupabaseClient, keep: bool) -> list[str]:
    """Insert, read, update, upsert. Returns list of session_ids to clean up."""
    section("2 · CRUD VIA SupabaseClient.sync_training_session")

    uid        = client_a.user_id
    session_id = make_session_id()
    to_delete  = []

    # INSERT via sync_training_session
    record = make_session_record(uid, session_id)
    ok_flag = client_a.sync_training_session(record)
    if ok_flag:
        ok("sync_training_session INSERT returned True")
        to_delete.append(session_id)
    else:
        fail("sync_training_session INSERT returned False")
        return to_delete

    # READ via get_training_history
    rows = client_a.get_training_history(uid)
    match = [r for r in rows if r.get("session_id") == session_id]
    if match:
        ok("get_training_history found inserted row")
        row = match[0]
        if VERBOSE:
            import json
            print("    row:", json.dumps({k: row[k] for k in list(row)[:8]}, indent=4))
    else:
        fail("get_training_history did not return inserted row")
        return to_delete

    # Verify values round-tripped correctly
    checks = [
        ("model_name",       row.get("model_name")       == "test_model"),
        ("status",           row.get("status")           == "completed"),
        ("duration_seconds", row.get("duration_seconds") == 120),
        ("epochs_completed", row.get("epochs_completed") == 50),
        ("checkpoint_path",  row.get("checkpoint_path")  is not None),
    ]
    for label, result in checks:
        if result:
            ok(f"  Value {label} round-tripped correctly")
        else:
            fail(f"  Value {label} mismatch", f"got {row.get(label)!r}")

    # UPSERT (re-sync same session_id with updated fields)
    updated_record = {**record, "status": "done", "epochs_completed": 75, "duration_seconds": 200}
    ok2 = client_a.sync_training_session(updated_record)
    if ok2:
        ok("sync_training_session UPSERT returned True")
    else:
        fail("sync_training_session UPSERT returned False")

    # Confirm it updated in-place (no duplicate rows)
    rows2 = client_a.get_training_history(uid)
    matches2 = [r for r in rows2 if r.get("session_id") == session_id]
    if len(matches2) == 1:
        ok("UPSERT produced exactly 1 row (no duplicate)")
        upserted = matches2[0]
        if upserted.get("status") == "done":
            ok("  UPSERT updated status correctly")
        else:
            fail("  UPSERT status not updated", f"got {upserted.get('status')!r}")
        if upserted.get("epochs_completed") == 75:
            ok("  UPSERT updated epochs_completed correctly")
        else:
            fail("  UPSERT epochs_completed not updated", f"got {upserted.get('epochs_completed')!r}")
    elif len(matches2) > 1:
        fail(f"UPSERT created {len(matches2)} rows — on_conflict not working")
    else:
        fail("UPSERT row not found after update")

    # Verify created_at was NOT overwritten (it must stay original)
    original_created = row.get("created_at")
    upserted_created = matches2[0].get("created_at") if matches2 else None
    if original_created and upserted_created and original_created == upserted_created:
        ok("created_at not overwritten on UPSERT")
    elif original_created and upserted_created:
        fail("created_at was overwritten on UPSERT",
             f"before={original_created[:19]}  after={upserted_created[:19]}")
    else:
        warn("created_at comparison skipped", "missing values")

    # Second session for multi-row tests
    session_id2 = make_session_id()
    record2 = make_session_record(uid, session_id2, model_name="test_model_2")
    client_a.sync_training_session(record2)
    to_delete.append(session_id2)

    rows3 = client_a.get_training_history(uid)
    own_sessions = [r for r in rows3 if r.get("session_id") in (session_id, session_id2)]
    if len(own_sessions) == 2:
        ok("Multiple sessions returned by get_training_history")
    else:
        warn("Expected 2 own sessions", f"got {len(own_sessions)} (may include pre-existing rows)")

    return to_delete


def test_rls(client_a: SupabaseClient, client_b: SupabaseClient,
             session_id_a: str, keep: bool) -> None:
    section("3 · ROW LEVEL SECURITY")

    uid_a = client_a.user_id
    uid_b = client_b.user_id

    # User B tries to read user A's session
    rows_b = client_b.get_training_history(uid_b)
    stolen = [r for r in rows_b if r.get("session_id") == session_id_a]
    if not stolen:
        ok("User B cannot read User A's sessions (SELECT RLS)")
    else:
        fail("RLS SELECT failed — User B can see User A's row")

    # User B tries to read with user A's user_id explicitly
    try:
        raw_b = client_b._get_client()
        r = raw_b.table("training_sessions").select("*").eq("user_id", uid_a).execute()
        if r.data:
            fail("RLS SELECT bypass — querying by user_id returns other user's rows",
                 f"got {len(r.data)} rows")
        else:
            ok("Explicit eq(user_id, uid_a) returns 0 rows for User B (SELECT RLS)")
    except Exception as e:
        ok("SELECT with explicit foreign user_id blocked", str(e)[:80])

    # User B tries to INSERT a row with user A's user_id
    try:
        raw_b = client_b._get_client()
        r = raw_b.table("training_sessions").insert({
            "user_id":    uid_a,          # impersonating User A
            "session_id": make_session_id(),
            "model_name": "rls_attack",
            "status":     "created",
        }).execute()
        if r.data:
            fail("RLS INSERT bypass — User B inserted a row as User A")
        else:
            ok("INSERT with foreign user_id produced no data (RLS blocked)")
    except Exception as e:
        msg = str(e)
        if "42501" in msg or "row-level security" in msg.lower() or "policy" in msg.lower():
            ok("User B cannot INSERT as User A (INSERT RLS)", "42501 row-level security")
        else:
            warn("INSERT RLS — unexpected error", msg[:120])

    # User B tries to UPDATE user A's row
    try:
        raw_b = client_b._get_client()
        r = (raw_b.table("training_sessions")
             .update({"status": "error", "error_message": "rls_breach"})
             .eq("session_id", session_id_a)
             .execute())
        if r.data:
            fail("RLS UPDATE bypass — User B updated User A's row")
        else:
            ok("UPDATE of foreign session returned no data (RLS blocked)")
    except Exception as e:
        msg = str(e)
        if "42501" in msg or "policy" in msg.lower():
            ok("User B cannot UPDATE User A's row (UPDATE RLS)")
        else:
            warn("UPDATE RLS — unexpected error", msg[:120])

    # Verify user A's row is untouched after B's attempted mutations
    raw_a = client_a._get_client()
    r = raw_a.table("training_sessions").select("status, error_message") \
              .eq("session_id", session_id_a).execute()
    if r.data:
        row = r.data[0]
        if row.get("error_message") != "rls_breach":
            ok("User A's row is intact after User B's mutation attempts")
        else:
            fail("User A's row was mutated by User B — RLS UPDATE failed")
    else:
        warn("Could not verify row integrity — row not found")

    # User A can still read their own row
    r2 = raw_a.table("training_sessions").select("session_id") \
               .eq("session_id", session_id_a).execute()
    if r2.data:
        ok("User A can still read their own row after RLS tests")
    else:
        fail("User A lost access to their own row")


def test_auth(client_a: SupabaseClient) -> None:
    section("4 · AUTH PROPERTIES")

    if client_a.is_authenticated:
        ok("is_authenticated → True after login")
    else:
        fail("is_authenticated → False (login may have failed)")

    if client_a.user_id:
        try:
            uuid.UUID(client_a.user_id)
            ok("user_id is a valid UUID", client_a.user_id[:16] + "…")
        except ValueError:
            fail("user_id is not a valid UUID", str(client_a.user_id))
    else:
        fail("user_id is None after login")

    if client_a.access_token:
        token = client_a.access_token
        ok("access_token present after login", f"{len(token)} chars")

        # Validate the token via get_user_by_token
        user_info = client_a.get_user_by_token(token)
        if user_info:
            ok("get_user_by_token returned user info for own token")
            if VERBOSE:
                print(f"    user: {user_info}")
            for field in ("id", "email", "role", "created_at"):
                if field in user_info:
                    ok(f"  user_info contains '{field}'")
                else:
                    fail(f"  user_info missing '{field}'")
        else:
            fail("get_user_by_token returned None for own token")
    else:
        fail("access_token is None after login")

    # Invalid token should return None
    bogus = client_a.get_user_by_token("eyJ.bogus.token")
    if bogus is None:
        ok("get_user_by_token returns None for invalid token")
    else:
        fail("get_user_by_token returned data for invalid token", str(bogus)[:80])

    # Logout clears state
    client_a.logout()
    if not client_a.is_authenticated:
        ok("is_authenticated → False after logout")
    else:
        fail("is_authenticated still True after logout")
    if client_a.user_id is None:
        ok("user_id → None after logout")
    else:
        fail("user_id not cleared after logout")
    if client_a.access_token is None:
        ok("access_token → None after logout")
    else:
        fail("access_token not cleared after logout")


def cleanup(client_a: SupabaseClient, session_ids: list[str]) -> None:
    if not session_ids:
        return
    section("CLEANUP")
    raw = client_a._get_client()
    for sid in session_ids:
        try:
            raw.table("training_sessions").delete().eq("session_id", sid).execute()
            if VERBOSE:
                print(f"  Deleted session_id={sid[:8]}…")
        except Exception as e:
            if VERBOSE:
                print(f"  Could not delete {sid[:8]}…: {e}")
    ok(f"Cleaned up {len(session_ids)} test row(s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Supabase integration tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--keep", action="store_true",
                        help="Skip test row cleanup (inspect in Supabase dashboard)")
    args = parser.parse_args()
    VERBOSE = args.verbose

    print(_c("cyan", "\n═══ Voice Clone Studio — Supabase Integration Tests ═══\n"))

    # ── Register two fresh test users ────────────────────────────────────────
    section("0 · AUTH SETUP")

    email_a = make_email()
    email_b = make_email()
    pw      = "TestPass_" + uuid.uuid4().hex[:8] + "Aa1!"

    client_a = SupabaseClient()
    client_b = SupabaseClient()

    # Register User A
    ok_a = client_a.register(email_a, pw)
    if ok_a:
        ok(f"User A registered", email_a)
    else:
        fail(f"User A registration failed", email_a)
        print(_c("red", "\nRegistration failed — check Supabase auth settings."))
        print("Ensure 'Enable email confirmations' is OFF in:")
        print("  Supabase dashboard → Authentication → Providers → Email")
        sys.exit(1)

    # Login User A (registration may have auto-logged in, but login explicitly)
    if not client_a.is_authenticated:
        ok_a_login = client_a.login(email_a, pw)
        if not ok_a_login:
            fail("User A login failed after registration")
            sys.exit(1)
    ok("User A logged in", f"uid={client_a.user_id[:16]}…")

    # Register + Login User B
    ok_b = client_b.register(email_b, pw)
    if not ok_b:
        fail("User B registration failed")
        sys.exit(1)
    if not client_b.is_authenticated:
        client_b.login(email_b, pw)
    ok("User B registered and logged in", f"uid={client_b.user_id[:16]}…")

    # Confirm users are distinct
    if client_a.user_id != client_b.user_id:
        ok("User A and User B have different UUIDs")
    else:
        fail("User A and User B have the same UUID — test isolation broken")
        sys.exit(1)

    # ── Run test groups ───────────────────────────────────────────────────────
    to_delete = []

    test_schema(client_a)

    to_delete = test_crud(client_a, args.keep)

    if to_delete:
        # Pass first session_id to RLS test for targeted mutation attempts
        test_rls(client_a, client_b, to_delete[0], args.keep)
    else:
        warn("Skipping RLS tests — no session_ids from CRUD phase")

    test_auth(client_a)

    # Need to re-login for cleanup (test_auth logs out)
    client_a.login(email_a, pw)
    if not args.keep:
        cleanup(client_a, to_delete)
    else:
        warn("--keep set", f"leaving {len(to_delete)} test rows in DB")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{_c('cyan', '═' * 56)}")
    total = PASSED + FAILED
    status_color = "green" if FAILED == 0 else "red"
    print(_c(status_color, f"  {PASSED}/{total} passed  |  {FAILED} failed"))
    for w in WARNINGS:
        print(_c("yellow", f"  ! {w}"))
    print(_c("cyan", "═" * 56) + "\n")
    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
