-- The fetch stage knows exactly what it downloaded (manual_subs / auto_subs /
-- audio), but had nowhere to record it, so the transcribe stage re-derived the
-- value from the filename with a heuristic that got it wrong: auto-generated
-- captions were published as `transcript_source: manual_subs`.
--
-- Provenance must be carried, not guessed.
ALTER TABLE artifacts ADD COLUMN meta TEXT;
