# ytdigest — agent notes

Canonical per-project instructions. `CLAUDE.md` is a symlink to this file, so
Claude Code, Codex, Kimi and Qwen all read the same thing.

**Read `README.md` for the design rationale.** The original architecture
specification is kept outside this repo; where it specifies a schema, an
interface or a path, implement exactly that.

## What this is

Watches YouTube channels, transcribes new videos (Cantonese/English
code-switched finance content), and publishes verified summaries as markdown
plus a Telegram digest.

## The one rule that shapes everything

**Wrong numbers are the only unacceptable failure.** A missing summary costs
nothing. A summary that says "revenue grew 23%" when the speaker said 13% is
actively harmful, because these summaries inform financial judgements.

That is why the pipeline has a deterministic `number_ledger` extracted by regex
*before* any LLM sees the transcript, and a `validator.py` that runs after
generation. The ledger is the authority; the LLM is not.

## Machine constraints that are not negotiable

- **16 GB unified memory**, shared with a dozen other background
  services on the same machine. This is why the summariser is a remote API and not a local MLX
  model, and why each stage runs as its own subprocess — MLX does not reliably
  return unified memory to the OS within a process.
- Sequential only. Two models in flight will thrash and end up slower.

## Verified environment facts

- `mlx-community/Qwen3-ASR-1.7B-8bit` exists on HuggingFace (checked
  2026-07-25). The 0.6B variant also exists if the 1.7B proves too slow.
- DeepSeek serves `deepseek-v4-flash` and `deepseek-v4-pro`. **Not**
  `deepseek-chat` / `deepseek-reasoner` — those IDs 404. Always hit
  `GET /models` rather than hardcoding from memory.
- **`v4-flash` follows the glossary rules better than `v4-pro`.** On a
  code-switched fixture, flash preserved `free cash flow` / `operating margin`
  / `payout ratio` / `holdco discount`; pro translated all four into Chinese.
  Flash is the default; pro is the escalation model for validator retries only,
  and its prompt needs the do-not-translate list restated.

## Findings that constrain the design

- **Both DeepSeek models normalise Chinese numerals unprompted**:
  百分之十三 → 13%, 四成 → 40%, 三成 → 30%. The validator therefore must compare
  against `number_ledger.normalized`, never `raw_text`, and the Chinese-numeral
  parser (億/萬/千萬/成/分之) is load-bearing, not a nice-to-have.
- `sqlite3.executescript()` issues an implicit COMMIT and will silently break
  out of an explicit transaction — schema applied, migration unrecorded. Use
  `db.split_statements()` instead. Do not "simplify" this back.
- **YouTube rolling auto-captions repeat text across cues.** Three approaches
  were tried on a real 2.5h file (`MgN00MCDDRM`, 9074s):
  | approach | chars | result |
  |---|---|---|
  | naive string dedup | 68,308 | clauses visible 2-3x |
  | key every char by inline timestamp | 102,463 | *worse* — carry-over gets a fresh cue-start key each time |
  | **line-aware + plain-cue recovery** | **35,882** | **0.3% repeat** |

  The working model: within a cue, only the line bearing inline `<ts><c>` tags
  is new; every line above it is carry-over and must be dropped. Cues with no
  inline timings anywhere are folded back only if their text (after its own
  prefix-trim pass) is absent from the timed reconstruction — otherwise short
  standalone utterances are silently lost. `tests/test_vtt.py` pins all of
  this; the fixture is hand-built because none of it is checkable by eye.
- **`live_chat` appears in yt-dlp's `subtitles` dict** alongside real tracks. It
  is a chat replay, not a transcript. `vtt.is_usable_subtitle_track()` blocks
  it — without that, "prefer manual subs" yields a transcript of viewer
  comments that looks plausible until you read it.
- Auto-captions carry **no confidence scores**. `Segment.confidence` stays
  `None` for them. Do not default it to 1.0 to make downstream code simpler —
  the validator would then treat unverified figures as verified.

## Conventions

- Filenames are **always** `<video_id>`. Titles contain CJK, emoji, slashes and
  200 characters of clickbait; they live in the DB and in frontmatter, never in
  a path.
- Colloquial Cantonese (嘅/咗/喺/唔) is preserved in transcripts, never
  normalised to standard written Chinese. It changes meaning.
- Terms in `config/glossary.yaml` stay in English everywhere.
- Secrets live in `.env` (chmod 600, gitignored). Never in `config.toml`.

## Commands

```sh
.venv/bin/ytdigest status          # queue, oldest pending, abandoned items
.venv/bin/ytdigest doctor          # run this first when something is wrong
.venv/bin/ytdigest channel add <url>
```

## Build order

M0/M1/M3/M4 are done; M2 (ASR) is deliberately unbuilt — see `STATUS.md` for
the decision rule. Do not build stages speculatively ahead of need.
