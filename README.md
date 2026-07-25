# ytdigest

Watches YouTube finance channels, transcribes new videos, and publishes
summaries **where every figure has been checked against the transcript**.

Built for Cantonese-language Hong Kong market commentary, which code-switches
constantly — Cantonese matrix with English financial terms and tickers embedded
mid-sentence.

## The problem it solves

A missing summary costs nothing. A summary that confidently says *"revenue grew
23%"* when the speaker said 13% is actively harmful, because these summaries
inform financial judgements.

That asymmetry drives the whole design. Before any LLM sees the transcript, a
deterministic pass extracts every figure into a **number ledger** with its unit,
timestamp and surrounding context. After generation, a **validator** checks
every number in the output against that ledger, matching on value *and* unit.
Unmatched figures trigger one regeneration naming the offenders; anything still
unverifiable is published marked `⚠︎` with a warning block, never as plain fact.

This is not theoretical. In live runs the model has:

- **invented** an index target (`28243`) when the captions truncated mid-sentence
  (`所以應該係八、二…`) — the prompt already forbade this;
- had a correct figure **wrongly rejected** by a naive string check, because the
  transcript wrote it in Chinese numerals (`二千億`, not `2000億`);
- had a real figure **split from its unit** by caption corruption
  (`已經有4200裏邊咧…其億美金`).

Prompt discipline caught none of these. The ledger catches all three.

## Pipeline

```
RSS → fetch → transcribe → normalize → summarize → validate → publish
                                          ↑            │
                                    number ledger ─────┘
```

Six stages, one direction. SQLite is a durable state machine, so any stage can
crash or be killed and re-running never redoes completed work. Each stage runs
as a **separate subprocess** — models do not reliably return memory to the OS
within a process. A `flock` makes a scheduled run during a long job a no-op.

## Install

```sh
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env && chmod 600 .env   # add your DeepSeek key
ytdigest doctor                          # run this first when anything is wrong
```

## Use

```sh
ytdigest channel add https://www.youtube.com/@SomeFinanceChannel
ytdigest add https://www.youtube.com/watch?v=VIDEO_ID   # one-off
ytdigest run                                            # what the scheduler calls
ytdigest status                                         # queue, stale items, failures
```

Telegram delivery is optional. Bot creation is the one step that cannot be
automated (Telegram has no API for it — @BotFather is a human-only chat), but
everything after is:

```sh
ytdigest telegram-setup --token <token-from-BotFather>
```

That verifies the token, waits for you to message the bot, then discovers and
stores the chat id itself.

## Scheduling

`ops/com.ytdigest.plist.template` is a launchd agent — not cron, which silently
skips runs when the machine sleeps. It wraps the run in `caffeinate -i` and
upgrades yt-dlp first, since a stale yt-dlp is the single most likely cause of a
silent pipeline stall.

## Design notes

- **Transcription is captions-first.** On a 2h31m Cantonese livestream, YouTube's
  auto-captions gave 100% time coverage with zero gaps >5s. ASR is designed for
  behind the `ASREngine` protocol but deliberately not built: it would fix two of
  four observed failure modes, while the validator catches all four. Better input
  lowers the flag rate; it does not change whether output is safe.
- **Rolling captions need real dedup.** YouTube re-displays the tail of each cue
  before appending new words. Naive string matching leaves clauses duplicated
  2–3×; keying on inline timestamps is *worse* (carry-over gets a fresh key each
  redisplay). Only the line-aware approach in `vtt.py` works. It has a hand-built
  fixture test because none of it is checkable by eye.
- **`live_chat` is not a transcript.** It appears alongside real subtitle tracks
  in yt-dlp output and is a chat replay. Including it yields a plausible-looking
  "transcript" of viewer comments.
- **Colloquial Cantonese is preserved.** 嘅 咗 喺 唔 哋 carry meaning; normalising
  them to written Chinese loses information about what was said.
- **Terms in `config/glossary.yaml` stay in English** in both transcript and
  summary.

## Status

See `STATUS.md`. Working end to end; the number ledger is the weakest component
and the one everything else depends on.

## Licence

MIT
