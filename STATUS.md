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

## Known gaps

- **Ledger coverage drives false flags.** `2022年` failed because the speaker
  said "2022" without 年. Several unverified figures are gaps of this kind, not
  fabrications. Widening the ledger is the highest-value next work.
- **No confidence scores on the captions path.** `Segment.confidence` is `None`
  and deliberately not faked, so the spec's "low-confidence figures always render
  ⚠︎" rule has no input. Only ASR restores that signal.
- **Chinese-numeral bare figures are not ledgered** (一/十 are too common in
  prose), so a summary written as `二萬億` cannot verify even when correct.
- Map-reduce is implemented but has not been exercised — the 2h31m transcript
  fits a single pass.

## To start it running

```sh
sed "s|__YTDIGEST_HOME__|$PWD|g" ops/com.ytdigest.plist.template \
  > ~/Library/LaunchAgents/com.ytdigest.plist
launchctl load ~/Library/LaunchAgents/com.ytdigest.plist
```

Runs 07:00 and 19:00 HKT under `caffeinate -i`, upgrading yt-dlp first.
Deliberately not loaded — starting a recurring job is the owner's call.
