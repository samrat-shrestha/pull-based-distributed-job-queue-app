-- pgcrypto provides gen_random_uuid(); create it before the table uses it.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS jobs (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- ---- request side (what the client submits) ----
  app_id            TEXT NOT NULL DEFAULT 'unknown',   -- who submitted (attribution/fairness)
  model             TEXT NOT NULL,                     -- e.g. 'qwen2.5:0.5b', 'flood-predict'
  task              TEXT,                              -- e.g. 'summarize', 'flood-predict' (worker routes on this)
  input             JSONB NOT NULL,                    -- the actual payload: {"prompt": "..."} or {"lat":.., "lon":..}
  params            JSONB NOT NULL DEFAULT '{}',       -- knobs: {"temperature":0.7, "max_tokens":512}
  response_type     TEXT NOT NULL DEFAULT 'sync',      -- 'sync' | 'async' | 'stream'
  priority          SMALLINT NOT NULL DEFAULT 0,       -- higher = claimed first

  -- ---- state side (managed by the broker/worker) ----
  status            TEXT NOT NULL DEFAULT 'queued',
  -- queued | running | done | failed
  worker_id         TEXT,
  lease_expires_at  TIMESTAMPTZ,
  result            JSONB,                             -- worker output (JSONB so it can hold structured results)
  error             TEXT,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at        TIMESTAMPTZ,
  finished_at       TIMESTAMPTZ
);

-- claim query orders by priority DESC, created_at ASC among queued jobs
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs (priority DESC, created_at ASC)
  WHERE status = 'queued';

-- lease sweeper looks up running jobs whose lease expired
CREATE INDEX IF NOT EXISTS jobs_lease_idx ON jobs (lease_expires_at)
  WHERE status = 'running';

-- attribution / history lookups per app
CREATE INDEX IF NOT EXISTS jobs_app_idx ON jobs (app_id, created_at DESC);
