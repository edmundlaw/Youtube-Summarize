# ytdigest — status

Last updated: 2026-07-26

## Working end to end

`ytdigest run` drives discover → fetch → transcribe → normalize → summarize →
publish, sequentially, one subprocess per stage, under a flock. Verified on a
real 55-minute Cantonese livestream: **processed 1, completed 1, failed 0**,
markdown written and digest delivered to Telegram.

| Milestone | State |
|---|---|
| M0 skeleton, DB, migrations, CLI, logging, `status`, `doctor` | done |
| M1 discover + fetch (subtitles) + publish | done |
| M2 ASR (Qwen3 via MLX) | **not built — deliberately** (see below) |
| M3 normalize + number ledger | done |
| M4 summarize + validator retry loop | done |
| M5 launchd, retries, notifications | plist written, **not loaded** |

47 tests, pyflakes clean, ~3,100 lines.

## Measured on the first real video

144 figures checked: **75.7% verified clean, 15.3% flagged, 9.0% unverified.**
Nothing unverified is published as fact — it is rendered `⚠︎` with a warning
block at the top of the document and a note in the Telegram message.

## Why M2 is not built

Auto-captions gave 100% time coverage on a 2h31m video (0 gaps >5s) and 0.3%
residual repetition after dedup. ASR would fix two of four observed failure
modes; the validator catches all four. Better input lowers the flag rate, it
does not change whether output is safe — so the control point was built first.

**The decision rule:** if the flag rate across ~10 videos stays near the 9%
measured here and the unverified items are mostly ledger gaps rather than model
fabrications, M2 never gets built. If it climbs, there is now a validator to
prove ASR actually helps rather than assuming it.

## External review, 2026-07-25

A cloud code review (Opus 5, clean checkout) found six defect classes. Every
claim was reproduced locally before any fix. Fixed and pinned by regression
tests in this commit:

| | defect | consequence |
|---|---|---|
| P0 | compound numerals truncated (三成半→30, 兩萬五→20000, 三點五厘→5) | **false PASS** — wrong number published as verified |
| P0b | index levels in Chinese numerals classed as clock times, which skip validation entirely | **false PASS** |
| P1 | annotation rewrote 1200億 as ⚠︎1⚠︎200億 | safety marker manufacturing a wrong number |
| P1b | killed stage left an orphan `running` row → no attempt count, no backoff, full-cost re-run forever | cost, indefinitely |
| P3 | MAGNITUDES ordered 萬 before 千萬 → 5千萬 unparseable; 千蚊 became a phantom exact 1000 | false alarms + one false-PASS vector |
| P5 | SchemaVersionError retried 3x; config prompt_version drifted from code | minor |

## Known gaps

- **Ledger coverage drives false flags.** `2022年` failed because the speaker
  said "2022" without 年. Several unverified figures are gaps of this kind, not
  fabrications. Widening the ledger is the highest-value next work.
- **No confidence scores on the captions path.** `Segment.confidence` is `None`
  and deliberately not faked, so the spec's "low-confidence figures always render
  ⚠︎" rule has no input. Only ASR restores that signal.
- **Chinese-numeral bare figures are not ledgered** (一/十 are too common in
  prose), so a summary written as `二萬億` cannot verify even when correct.
- **Validation has no locality.** `verified` means the value+unit was spoken
  somewhere in the video, not that it was spoken about *this* claim. The
  speaker half of this is now constrained (host roster, below); the subject
  half is not — a real 4000 spoken about gold can still be attached to the
  wrong instrument and verify clean.
- **Speaker attribution is constrained but not validated.** The prompt is given
  the host list and told trailers name outsiders; nothing checks the output
  against it the way the ledger checks numbers.
- **Error text is persisted and forwarded unredacted.** `logging.py` scrubs
  structlog events, but `stage_runs.error_text` is written raw, printed by
  `status`, and 80 chars of it are sent to Telegram. Not yet fixed.
- Publish sends Telegram before committing, so a kill in that window
  double-delivers. Not yet fixed.

## To start it running

```sh
sed "s|__YTDIGEST_HOME__|$PWD|g" ops/com.ytdigest.plist.template \
  > ~/Library/LaunchAgents/com.ytdigest.plist
launchctl load ~/Library/LaunchAgents/com.ytdigest.plist
```

Runs 07:00 and 19:00 HKT under `caffeinate -i`, upgrading yt-dlp first.
Deliberately not loaded — starting a recurring job is the owner's call.
