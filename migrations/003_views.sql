-- Market views expressed in videos: the durable, queryable record.
--
-- Design decisions, and why:
--
-- * One row per (speaker, instrument, call). A single video yields many rows.
--   A view is the unit you will backtest, not a video and not a summary.
--
-- * `stated_at` is an absolute UTC timestamp, not an offset into the video.
--   Backtesting and charting both need a real time axis; an offset is useless
--   without joining back to the video every time. `start_s` is kept alongside
--   so you can still jump to the moment in the recording.
--
-- * `instrument` is a NORMALISED symbol (0700.HK, ^HSI, USDJPY, CL=F, XAUUSD)
--   and `instrument_raw` is what was actually said (騰訊, 恒指, 布油, 金).
--   Without the normalised form you cannot join to price data; without the raw
--   form you cannot audit a bad mapping. Both are needed.
--
-- * `level_value` is NUMERIC and `ledger_id` points at the number_ledger row it
--   came from. A level that cannot be traced to a verified figure is the exact
--   failure this project exists to prevent, so provenance is a column, not a
--   convention. A view with no level (a directional opinion) is still valid —
--   level_value is nullable.
--
-- * Outcome columns are nullable and filled later by a resolver that compares
--   the level against real prices. Storing them here rather than in a separate
--   table keeps the backtest query a single scan.
--
-- * No prices are stored. Price history belongs in whatever market-data source
--   you already run; duplicating it here would go stale silently.

CREATE TABLE instruments (
    symbol       TEXT PRIMARY KEY,      -- normalised: 0700.HK, ^HSI, USDJPY
    asset_class  TEXT NOT NULL,         -- equity|index|fx|commodity|crypto|rate|macro
    display_name TEXT,
    currency     TEXT,                  -- quote currency, for level sanity checks
    added_at     TEXT NOT NULL
);

-- Spoken aliases. Cantonese finance uses nicknames heavily (鵝廠 = Tencent,
-- 大笨象 = HSBC), and the same instrument is named differently by each host.
CREATE TABLE instrument_aliases (
    alias     TEXT PRIMARY KEY,
    symbol    TEXT NOT NULL REFERENCES instruments(symbol),
    added_at  TEXT NOT NULL
);

CREATE TABLE views (
    id              INTEGER PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id),
    channel_id      TEXT NOT NULL REFERENCES channels(id),

    -- who and when
    speaker         TEXT,               -- must come from the video's host roster
    stated_at       TEXT NOT NULL,      -- absolute UTC: video publish + start_s
    start_s         REAL NOT NULL,      -- offset into the recording

    -- what it is about
    instrument      TEXT REFERENCES instruments(symbol),   -- NULL = unmapped
    instrument_raw  TEXT NOT NULL,      -- exactly as spoken
    asset_class     TEXT,

    -- the call
    direction       TEXT NOT NULL,      -- long|short|neutral|avoid|exit
    conviction      TEXT,               -- high|medium|low
    thesis          TEXT NOT NULL,      -- the claim, in the speaker's terms
    reasoning       TEXT,

    -- the number, if one was given
    level_type      TEXT,               -- target|support|resistance|entry|stop|valuation
    level_value     REAL,
    level_unit      TEXT,               -- hkd|usd|cny|pct|points|multiple
    ledger_id       INTEGER REFERENCES number_ledger(id),  -- provenance
    level_verified  INTEGER NOT NULL DEFAULT 0,            -- 1 only if matched

    -- when it should come true
    horizon         TEXT,               -- intraday|days|weeks|months|quarters|year
    horizon_ends_at TEXT,               -- absolute, when derivable

    -- filled in later by the resolver
    outcome         TEXT,               -- pending|hit|missed|expired|void
    outcome_value   REAL,               -- the price that settled it
    outcome_note    TEXT,
    resolved_at     TEXT,

    -- lineage back to the summary that produced this row
    summary_id      INTEGER REFERENCES summaries(id),
    prompt_version  TEXT,
    created_at      TEXT NOT NULL
);

-- The three access patterns this table exists for.
CREATE INDEX idx_views_instrument ON views(instrument, stated_at);   -- charting
CREATE INDEX idx_views_speaker    ON views(speaker, stated_at);      -- track record
CREATE INDEX idx_views_pending    ON views(outcome, horizon_ends_at);-- resolver sweep
CREATE INDEX idx_views_video      ON views(video_id);

-- A speaker cannot make the identical call twice in one video; re-running the
-- summariser must update rather than duplicate.
CREATE UNIQUE INDEX idx_views_dedupe
    ON views(video_id, speaker, instrument_raw, direction, IFNULL(level_value, -1));
