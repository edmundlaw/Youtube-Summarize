-- ytdigest initial schema. See ARCHITECTURE.md §4.
-- Blobs never live here; only metadata, status and lineage.

CREATE TABLE channels (
    id              TEXT PRIMARY KEY,      -- UC... channel id
    title           TEXT NOT NULL,
    lang_hint       TEXT,                  -- 'yue' | 'zh' | 'en' | NULL (auto)
    enabled         INTEGER NOT NULL DEFAULT 1,
    min_duration_s  INTEGER DEFAULT 180,   -- skip Shorts and clips
    max_duration_s  INTEGER DEFAULT 10800,
    title_include   TEXT,                  -- optional regex
    title_exclude   TEXT,
    added_at        TEXT NOT NULL
);

CREATE TABLE videos (
    id             TEXT PRIMARY KEY,       -- YouTube video id, canonical key everywhere
    channel_id     TEXT NOT NULL REFERENCES channels(id),
    title          TEXT NOT NULL,
    published_at   TEXT NOT NULL,
    duration_s     INTEGER,
    discovered_at  TEXT NOT NULL,
    status         TEXT NOT NULL           -- see state machine
);
CREATE INDEX idx_videos_status ON videos(status, published_at);

CREATE TABLE stage_runs (
    id            INTEGER PRIMARY KEY,
    video_id      TEXT NOT NULL REFERENCES videos(id),
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,           -- running | ok | failed
    attempt       INTEGER NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    duration_ms   INTEGER,
    error_class   TEXT,                    -- retryable | permanent
    error_text    TEXT,
    next_retry_at TEXT
);
CREATE INDEX idx_stage_runs_video ON stage_runs(video_id, stage);

CREATE TABLE artifacts (
    video_id    TEXT NOT NULL REFERENCES videos(id),
    kind        TEXT NOT NULL,             -- audio | transcript | normalized | summary
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    bytes       INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (video_id, kind)
);

CREATE TABLE transcripts (
    video_id        TEXT PRIMARY KEY REFERENCES videos(id),
    schema_version  INTEGER NOT NULL,
    source          TEXT NOT NULL,         -- asr | manual_subs | auto_subs
    model_id        TEXT,
    params_hash     TEXT,                  -- hash of decode params, for cache invalidation
    dominant_lang   TEXT,
    segment_count   INTEGER,
    mean_confidence REAL,
    created_at      TEXT NOT NULL
);

CREATE TABLE number_ledger (
    id           INTEGER PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES videos(id),
    raw_text     TEXT NOT NULL,            -- exactly as it appears in transcript
    normalized   TEXT,                     -- 13.5 | 2.3e9 | NULL if unparseable
    unit         TEXT,                     -- pct | hkd | usd | cny | multiple | count | year | bps
    segment_id   INTEGER NOT NULL,
    start_s      REAL NOT NULL,
    confidence   REAL,
    context      TEXT NOT NULL             -- +/-1 segment of surrounding text
);
CREATE INDEX idx_ledger_video ON number_ledger(video_id);

CREATE TABLE summaries (
    id              INTEGER PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id),
    prompt_version  TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    transcript_sha  TEXT NOT NULL,         -- lineage: which transcript produced this
    validator_state TEXT NOT NULL,         -- passed | passed_with_flags | failed
    path            TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_summaries_video ON summaries(video_id, created_at);
