-- Run history for the speech emotion pipeline.
--
-- Paste this into the Supabase SQL editor (Dashboard -> SQL Editor -> New
-- query -> Run). One table, two indexes, RLS on.
--
-- Why RLS is enabled with NO policies: that denies every request made with
-- the anon/publishable key, which is the key that can end up in a browser.
-- The secret key bypasses RLS, and only the server-side Vercel route holds
-- it -- so the table is reachable from the API route and from nowhere else.
-- Enabling RLS and forgetting the policies is the safe failure here; the
-- unsafe one would be leaving RLS off.

create table if not exists public.analysis_runs (
  id                    uuid primary key default gen_random_uuid(),
  created_at            timestamptz not null default now(),

  -- Who ran it. From the Auth.js session server-side, never from the client,
  -- so a caller cannot attribute a run to someone else.
  user_email            text not null,

  -- What was analysed.
  source_kind           text not null
                          check (source_kind in ('corpus', 'upload', 'recording')),
  source                text not null,   -- clip key, or the file name
  clip_key              text,            -- set only for corpus clips
  duration_sec          real not null,

  -- What the model said.
  predicted_emotion     text not null,
  predicted_sentiment   text not null,
  emotion_confidence    real not null,
  sentiment_confidence  real not null,

  -- Ground truth, present only for corpus clips.
  gold_emotion          text,
  gold_sentiment        text,
  emotion_correct       boolean,
  sentiment_correct     boolean,

  -- Diagnostics.
  transcript            text not null,
  reference_text        text,
  wer_normalised        real,
  wer_exact             real,
  speech_ratio          real not null,
  gate_passes           boolean,

  -- Timings. server_ms is what the endpoint spent on the audio; wall_ms is
  -- what the user waited, which on a cold start includes container start.
  server_ms             integer not null,
  wall_ms               integer not null,
  cold_start            boolean not null,

  -- The whole response, so a row can reopen the full result panel without
  -- the schema above having to grow every time the endpoint returns more.
  result                jsonb not null
);

create index if not exists analysis_runs_created_at_idx
  on public.analysis_runs (created_at desc);

create index if not exists analysis_runs_user_email_idx
  on public.analysis_runs (user_email);

alter table public.analysis_runs enable row level security;

-- Deliberately no policies. See the note at the top.
