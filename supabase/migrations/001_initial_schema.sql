-- ============================================================
-- Voice Clone Studio — Initial Schema
-- Migration: 001_initial_schema
-- Safe to run multiple times (all statements use IF NOT EXISTS / OR REPLACE)
-- ============================================================


-- ─── Training Sessions ───────────────────────────────────────────────────────
--
-- Central table for all training runs.
-- Primary state lives in local JSON (session_manager.py); this table is
-- the remote mirror synced at pipeline completion and queryable via the API.
--
-- Column notes
--   session_id       — mirrors the UUID used in session_manager.py; UNIQUE so
--                      the upsert conflict target is unambiguous.
--   user_id          — FK to auth.users; CASCADE so deleting a Supabase user
--                      removes all their training history automatically.
--   status           — includes both local states AND "completed" which is what
--                      backend.py sends in the sync payload (do not remove it).
--   files / hyperparams / hardware_profile — JSONB; nullable since summary-only
--                      syncs do not include these fields yet.
--   segment_manifest — local filesystem path, nullable by design.
--   created_at       — immutable after INSERT; updated_at tracks mutations.

CREATE TABLE IF NOT EXISTS public.training_sessions (

    -- Surrogate PK
    id               UUID        NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,

    -- Owning user (hard FK — no orphaned rows)
    user_id          UUID        NOT NULL
                     REFERENCES  auth.users(id) ON DELETE CASCADE,

    -- Natural key matching local session_manager UUIDs; the upsert conflict target
    session_id       TEXT        NOT NULL UNIQUE,

    -- Model info
    model_name       TEXT        NOT NULL,

    -- Full pipeline lifecycle states from session_manager.py
    -- plus "completed" used by backend.py's sync payload
    status           TEXT        NOT NULL DEFAULT 'created'
                     CHECK (status IN (
                         'created',
                         'preprocessing',
                         'preprocessing_done',
                         'awaiting_confirmation',
                         'training',
                         'paused',
                         'done',
                         'completed',   -- backend.py sync payload value
                         'error'
                     )),

    -- Input files (array of absolute paths on the training machine)
    files            JSONB                DEFAULT '[]'::jsonb,

    -- LLM-generated or user-overridden hyperparameters
    hyperparams      JSONB                DEFAULT '{}'::jsonb,

    -- Live training progress (populated during training, NULL for completed syncs)
    current_epoch    INTEGER              DEFAULT 0
                     CHECK (current_epoch   >= 0),
    total_epochs     INTEGER              DEFAULT 0
                     CHECK (total_epochs    >= 0),
    latest_loss      DOUBLE PRECISION,

    -- Completion summary (populated by sync_training_session)
    epochs_completed INTEGER              DEFAULT 0
                     CHECK (epochs_completed >= 0),
    duration_seconds INTEGER              DEFAULT 0
                     CHECK (duration_seconds >= 0),
    hardware_profile JSONB                DEFAULT '{}'::jsonb,

    -- Outputs
    checkpoint_path  TEXT,
    segment_manifest TEXT,     -- local path; not useful remotely but kept for round-trip

    -- Error info
    error_message    TEXT,

    -- Immutable creation time; updated_at maintained by trigger below
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ─── Auto-update trigger for updated_at ──────────────────────────────────────

CREATE OR REPLACE FUNCTION public.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_training_sessions_updated_at
    ON public.training_sessions;

CREATE TRIGGER trg_training_sessions_updated_at
    BEFORE UPDATE ON public.training_sessions
    FOR EACH ROW
    EXECUTE FUNCTION public.fn_set_updated_at();


-- ─── Indexes ─────────────────────────────────────────────────────────────────
-- All queries in the backend filter by user_id; compound index covers
-- get_training_history(user_id) and find_by_model(user_id, model_name).

CREATE INDEX IF NOT EXISTS idx_ts_user_id
    ON public.training_sessions (user_id);

CREATE INDEX IF NOT EXISTS idx_ts_status
    ON public.training_sessions (status);

CREATE INDEX IF NOT EXISTS idx_ts_user_model
    ON public.training_sessions (user_id, model_name);


-- ─── Row Level Security ───────────────────────────────────────────────────────
-- Every row is owned by exactly one auth.users row.
-- auth.uid() returns the UUID from the JWT presented with the request.
-- The service role key bypasses RLS entirely (used for admin/migrations only).

ALTER TABLE public.training_sessions ENABLE ROW LEVEL SECURITY;

-- SELECT: users see only their own rows
DROP POLICY IF EXISTS "ts_select_own" ON public.training_sessions;
CREATE POLICY "ts_select_own"
    ON public.training_sessions
    FOR SELECT
    USING (auth.uid() = user_id);

-- INSERT: users can only insert rows they own
DROP POLICY IF EXISTS "ts_insert_own" ON public.training_sessions;
CREATE POLICY "ts_insert_own"
    ON public.training_sessions
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- UPDATE: users can only mutate their own rows
DROP POLICY IF EXISTS "ts_update_own" ON public.training_sessions;
CREATE POLICY "ts_update_own"
    ON public.training_sessions
    FOR UPDATE
    USING    (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- DELETE: blocked via API — service role can still delete directly
-- (no policy = no access for authenticated/anon roles)


-- ─── Grant PostgREST access ───────────────────────────────────────────────────
-- PostgREST runs as the anon/authenticated role; it needs explicit GRANT to
-- access the table and the trigger function.

GRANT SELECT, INSERT, UPDATE ON public.training_sessions
    TO anon, authenticated;

GRANT EXECUTE ON FUNCTION public.fn_set_updated_at()
    TO anon, authenticated;
