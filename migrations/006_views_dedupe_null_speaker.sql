-- The dedupe index used a bare `speaker` column, and SQLite treats NULLs in a
-- UNIQUE index as distinct from each other. Every view whose speaker could not
-- be attributed therefore failed to conflict, so `views-reindex` inserted a
-- fresh copy of it on each run instead of updating in place.
--
-- The damage is not just untidy rows. The scorecard counts by outcome, so each
-- reindex silently inflated the '(unattributed)' bucket -- 84 views became 168
-- after one run -- and any hit rate computed over a duplicated denominator is
-- wrong in a way that looks entirely plausible.
--
-- Collapse to the earliest row of each group, then rebuild the index over
-- IFNULL(speaker,'') so unattributed views compare equal to one another.

DELETE FROM views
WHERE id NOT IN (
    SELECT MIN(id) FROM views
    GROUP BY video_id, IFNULL(speaker, ''), instrument_raw, direction,
             IFNULL(level_value, -1)
);

DROP INDEX IF EXISTS idx_views_dedupe;

CREATE UNIQUE INDEX idx_views_dedupe
    ON views(video_id, IFNULL(speaker, ''), instrument_raw, direction,
             IFNULL(level_value, -1));
