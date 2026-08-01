-- Speaker identification by voice.
--
-- YouTube auto-captions carry no diarization. With three hosts talking over
-- each other, any speaker name in a summary was previously a guess by the
-- model -- and a guessed attribution is worse than no attribution, because the
-- whole point of the track record is judging named individuals.
--
-- The only place the information actually exists is the audio. A voiceprint is
-- a mean speaker embedding, built once per person from audio where we know for
-- certain who is talking (a solo video), then matched against each caption
-- segment thereafter.

CREATE TABLE voiceprints (
    speaker      TEXT PRIMARY KEY,      -- canonical name, as in people.yaml
    embedding    BLOB NOT NULL,         -- float32 vector, L2-normalised
    dim          INTEGER NOT NULL,
    model        TEXT NOT NULL,         -- embeddings are not comparable across
                                        -- models; a change invalidates the row
    n_clips      INTEGER NOT NULL,      -- how many windows were averaged
    total_s      REAL NOT NULL,         -- how much audio backs it
    source_note  TEXT,                  -- which videos it came from, for audit
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

-- Per-segment attribution, kept separately from `views` so that re-running
-- identification never rewrites the extracted claims themselves.
CREATE TABLE segment_speakers (
    video_id    TEXT NOT NULL REFERENCES videos(id),
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    speaker     TEXT,                   -- NULL = below threshold, unattributed
    score       REAL,                   -- cosine similarity to the best match
    margin      REAL,                   -- best minus runner-up; low = ambiguous
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (video_id, start_s)
);

CREATE INDEX idx_segment_speakers_video ON segment_speakers(video_id, start_s);

-- Which prompt version produced a view matters for the record: v2 and earlier
-- attributed speakers by guesswork, v3 refuses to, and voice-identified views
-- are the only ones that should carry real weight. `attribution` records how
-- the name on a view was arrived at, so the scorecard can separate them.
ALTER TABLE views ADD COLUMN attribution TEXT;   -- voice|cue|guessed|none

-- Everything already stored predates voice identification.
UPDATE views SET attribution = 'guessed' WHERE speaker IS NOT NULL;
UPDATE views SET attribution = 'none'    WHERE speaker IS NULL;
