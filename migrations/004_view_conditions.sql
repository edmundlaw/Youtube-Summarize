-- Most calls are conditional, and flattening them to a direction misrepresents
-- the speaker.
--
-- Observed: KC on SK Hynix said 「如果他彈的話 我覺得應該是沽的」 — "if it
-- bounces, then I think you sell". Stored as a bare `short`, a backtest would
-- enter at the moment he spoke, which is precisely what he said NOT to do. The
-- call only becomes live on a rally; if the stock kept falling it never
-- triggered and grading it as a short is meaningless.
--
-- Two columns, kept separate because they answer different questions:
--   entry_basis — a small controlled vocabulary, so a backtester can branch
--   condition   — the trigger in the speaker's own words, for audit
--
-- `stance` is the durable opinion (bearish here), `direction` stays the action.
-- A speaker can be structurally bearish while saying "don't short it yet".

ALTER TABLE views ADD COLUMN entry_basis TEXT;   -- immediate|on_rally|on_dip|
                                                 -- on_break|on_confirmation|unspecified
ALTER TABLE views ADD COLUMN condition TEXT;     -- verbatim trigger, if stated
ALTER TABLE views ADD COLUMN stance TEXT;        -- bullish|bearish|neutral

CREATE INDEX idx_views_basis ON views(entry_basis, outcome);
