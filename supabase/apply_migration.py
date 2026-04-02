#!/usr/bin/env python3
"""
Apply Voice Clone Studio Supabase migrations.

Requires ONE of the following in your .env (or environment):

  Option A — direct Postgres connection (recommended):
    DATABASE_URL=postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres

    Your password is in the Supabase dashboard:
    Project Settings → Database → Connection string → URI

  Option B — Supabase service role key (for Management API):
    SUPABASE_SERVICE_KEY=<service_role_key>

    Found in: Project Settings → API → service_role key

Usage:
    python supabase/apply_migration.py
    python supabase/apply_migration.py --dry-run      # print SQL only
    python supabase/apply_migration.py --check        # check current schema state
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Ordered list of migration files to apply
MIGRATIONS = [
    "001_initial_schema.sql",
]


def load_sql(filename: str) -> str:
    path = MIGRATIONS_DIR / filename
    with open(path) as f:
        return f.read()


# ── Option A: direct psycopg2 connection ─────────────────────────────────────

def apply_via_postgres(db_url: str, dry_run: bool = False) -> bool:
    try:
        import psycopg2
    except ImportError:
        print("[apply] psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

    print(f"[apply] Connecting via DATABASE_URL ...")
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()
    except Exception as e:
        print(f"[apply] Connection failed: {e}")
        return False

    for migration in MIGRATIONS:
        sql = load_sql(migration)
        print(f"\n[apply] {'[DRY-RUN] ' if dry_run else ''}Applying {migration} ...")
        if dry_run:
            print(sql)
            continue
        try:
            cur.execute(sql)
            print(f"[apply] {migration} — OK")
        except Exception as e:
            print(f"[apply] {migration} — FAILED: {e}")
            conn.close()
            return False

    cur.close()
    conn.close()
    print("\n[apply] All migrations applied successfully.")
    return True


# ── Option B: Supabase Management API ────────────────────────────────────────

def apply_via_management_api(
    project_ref: str,
    service_key: str,
    dry_run: bool = False,
) -> bool:
    try:
        import httpx
    except ImportError:
        print("[apply] httpx not installed. Run: pip install httpx")
        return False

    # The Management API SQL endpoint accepts service_role JWT
    endpoint = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"
    headers  = {
        "Authorization": f"Bearer {service_key}",
        "Content-Type":  "application/json",
    }

    for migration in MIGRATIONS:
        sql = load_sql(migration)
        print(f"\n[apply] {'[DRY-RUN] ' if dry_run else ''}Applying {migration} ...")
        if dry_run:
            print(sql)
            continue
        try:
            r = httpx.post(endpoint, headers=headers, json={"query": sql}, timeout=30)
            if r.status_code in (200, 201):
                print(f"[apply] {migration} — OK")
            else:
                print(f"[apply] {migration} — HTTP {r.status_code}: {r.text[:200]}")
                return False
        except Exception as e:
            print(f"[apply] {migration} — request failed: {e}")
            return False

    print("\n[apply] All migrations applied successfully.")
    return True


# ── Schema check (read-only) ──────────────────────────────────────────────────

def check_schema(db_url: str) -> None:
    try:
        import psycopg2
    except ImportError:
        print("[check] psycopg2 not installed. Run: pip install psycopg2-binary")
        return

    conn = psycopg2.connect(db_url)
    cur  = conn.cursor()

    # Check table exists
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND   table_name   = 'training_sessions';
    """)
    exists = cur.fetchone()
    print(f"training_sessions table: {'EXISTS' if exists else 'MISSING'}")

    if exists:
        # Column list
        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND   table_name   = 'training_sessions'
            ORDER BY ordinal_position;
        """)
        print("\nColumns:")
        for row in cur.fetchall():
            print(f"  {row[0]:25s}  {row[1]:20s}  nullable={row[2]:3s}  default={row[3]}")

        # RLS check
        cur.execute("""
            SELECT relrowsecurity
            FROM pg_class
            WHERE relname = 'training_sessions'
            AND   relnamespace = 'public'::regnamespace;
        """)
        rls = cur.fetchone()
        print(f"\nRLS enabled: {bool(rls and rls[0])}")

        # Policies
        cur.execute("""
            SELECT policyname, cmd, qual
            FROM pg_policies
            WHERE tablename = 'training_sessions'
            AND   schemaname = 'public';
        """)
        policies = cur.fetchall()
        print(f"\nPolicies ({len(policies)}):")
        for p in policies:
            print(f"  [{p[1]}] {p[0]}: {p[2]}")

        # Indexes
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'training_sessions'
            AND   schemaname = 'public';
        """)
        print("\nIndexes:")
        for row in cur.fetchall():
            print(f"  {row[0]}")

    cur.close()
    conn.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Apply Supabase migrations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print SQL without executing")
    parser.add_argument("--check",   action="store_true",
                        help="Check current schema state (requires DATABASE_URL)")
    args = parser.parse_args()

    # Load env
    env_path = ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    db_url      = os.getenv("DATABASE_URL", "")
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    supabase_url = os.getenv("SUPABASE_URL", "")
    project_ref = supabase_url.split("//")[1].split(".")[0] if "//" in supabase_url else ""

    if args.check:
        if not db_url:
            print("[check] DATABASE_URL required for schema check.")
            sys.exit(1)
        check_schema(db_url)
        return

    if db_url:
        ok = apply_via_postgres(db_url, dry_run=args.dry_run)
    elif service_key and project_ref:
        ok = apply_via_management_api(project_ref, service_key, dry_run=args.dry_run)
    else:
        # No credentials — print the SQL and instructions
        print("=" * 60)
        print("No DATABASE_URL or SUPABASE_SERVICE_KEY found.")
        print("Run the SQL below in the Supabase SQL Editor:")
        print(f"  {supabase_url.replace('supabase.co', 'supabase.com')}/project/{project_ref}/sql")
        print("=" * 60)
        for migration in MIGRATIONS:
            print(f"\n-- {migration}\n")
            print(load_sql(migration))
        print("\nTo apply automatically, add ONE of these to your .env:")
        print("  DATABASE_URL=postgresql://postgres:<password>@db."
              f"{project_ref}.supabase.co:5432/postgres")
        print("  SUPABASE_SERVICE_KEY=<your service_role key>")
        sys.exit(0)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
