---
schema: 2
# Front-matter created by `mc migrate-status`: this file
# had none, so it could not report its own status at all.
# CHECK status/progress/phase — they are a starting point.
status: BUILDING
progress: 85
phase: operating
summary: >
  Watches six Hong Kong finance channels, transcribes Cantonese/English
  code-switched commentary, and publishes verified summaries as markdown plus a
  Telegram digest. Running in production under launchd since 2026-07; 73 videos
  published. Every figure is extracted to a deterministic ledger before any LLM
  sees the transcript, and since 2026-08-12 is re-listened to by local ASR, so a
  mis-heard number is refused rather than published as verified.

# What has been BUILT, in plain language, <=100 chars each.
features:
  - Watches channels via RSS and queues new uploads, filtered by show and by which host is on it
  - SQLite state machine, one subprocess per stage, single-instance under flock
  - Deterministic number ledger extracted by regex before any LLM sees the transcript
  - Validator that refuses any summary figure it cannot match against that ledger
  - Speaker identification by voiceprint, so attribution is measured rather than guessed
  - ASR cross-check: local Qwen3 re-listens to every figure and disputes are refused
  - Markdown output plus a Telegram digest, suppressed for videos older than 3 days
  - Views database with price resolution, for scoring each host's calls over time

items: []
---
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
| M2 ASR (Qwen3 via MLX) | **spans mode done** 2026-08-12; full-file mode still unbuilt (see below) |
| M3 normalize + number ledger | done |
| M4 summarize + validator retry loop | done |
| M5 launchd, retries, notifications | loaded and running |

194 tests, pyflakes clean.

## Measured on the first real video

144 figures checked: **75.7% verified clean, 15.3% flagged, 9.0% unverified.**
Nothing unverified is published as fact — it is rendered `⚠︎` with a warning
block at the top of the document and a note in the Telegram message.

## M2, and why half of it is now built

**Resolved 2026-08-12.** The rule below asked whether a *better* transcript was
worth building, and the answer turned out to be the wrong question — the
captions are not merely noisier than ASR, they are wrong in ways nothing could
see. 中芯 appears 0 times in 5M characters of stored transcript against 288 for
中心; a figure the captions record as 29億 is heard by two independent models as
299億. The validator was blind to all of it by construction, checking a summary
against a ledger built from the same wrong transcript.

So ASR was built as a **cross-check rather than a replacement**: Qwen3-ASR
re-listens to the ~8-second window around every ledger figure, and a
disagreement is refused rather than resolved. Live since 2026-08-12. Measured on
a real 18-minute episode: 21 agreed, 53 absent, 9 disputed, of which at least
two are genuine caption errors (Tencent 「49呀幾」 against a discussion of 465蚊
— ASR hears 490; gold 「440前後」 — ASR hears 4400).

**Full-file mode is still unbuilt**, which is why 全職炒家 RON LAU still produces
nothing (see below). Qwen returns one untimed segment for its whole input, so a
whole-video pass would place every figure at t=0 and silently break speaker
identification. It needs VAD chunking first.

Two limits worth stating plainly: 53 of 83 figures came back *unchecked*, so
this is a net for the worst errors and not a guarantee; and precision is
untuned, so some flags are spurious. A flag means "these two disagree, go
listen", never "the second number is right".

The original reasoning, kept because the decision rule still governs full-file
mode:

Auto-captions gave 100% time coverage on a 2h31m video (0 gaps >5s) and 0.3%
residual repetition after dedup. ASR would fix two of four observed failure
modes; the validator catches all four. Better input lowers the flag rate, it
does not change whether output is safe — so the control point was built first.

**The decision rule:** if the flag rate across ~10 videos stays near the 9%
measured here and the unverified items are mostly ledger gaps rather than model
fabrications, M2 never gets built. If it climbs, there is now a validator to
prove ASR actually helps rather than assuming it.

**A second trigger appeared on 2026-08-09, and it is not about quality.**
全職炒家 RON LAU (`UCZGzrIUFtkSwidtKx7NH-Zg`) has **no captions at all** — ten
consecutive uploads checked, `automatic_captions` empty on every one. His live
streams carry a single `subtitles` track and it is `live_chat`, which
`vtt.is_usable_subtitle_track` correctly refuses. The only videos on the channel
that do have captions are the 50-second 紅綠燈AI模型 product ads.

So the flag rate is no longer the whole question. The rule above assumes a
transcript exists and asks whether a better one is worth building; this is a
channel that cannot be summarised at any quality without ASR. Every upload
fetches, then abandons at `transcribe`. The channel is left enabled rather than
disabled, so that the cost of not having M2 stays visible instead of quietly
disappearing from the queue.

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
- **Speaker attribution is now measured, but only for enrolled voices.**
  `voice.py` matches each caption segment against a voiceprint and `views.
  attribution` records how each name was obtained. Only 羅家聰 (KC) is enrolled
  so far; every other host's segments come back unattributed, which is honest
  but leaves them with no measurable record. Enrolling them needs one solo
  video each (check the title — 「嘉賓」 or an interview format disqualifies it),
  or ~60 seconds hand-labelled once.
- **Views stored before voice ID exist carry `attribution='guessed'`.** They
  were attributed by a model reading unlabelled captions and are not
  trustworthy for judging a person. The scorecard shows voice and guess counts
  separately; do not pool them. Re-running `identify` on a video and then
  `views-reindex` upgrades it.
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

Runs 22:30 and 06:30 HKT under `caffeinate -i`, upgrading yt-dlp first.
Both sit in DeepSeek's off-peak window (peak is 09:00-12:00 and 14:00-18:00
HKT, charged at double), so every run bills at the base rate.
Deliberately not loaded — starting a recurring job is the owner's call.
