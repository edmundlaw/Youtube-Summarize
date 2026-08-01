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

## Captions have no speaker labels — that is what `voice.py` is for

Auto-captions carry no diarization at all. On a three-host show the model had
no way to know who spoke, and the prompt made it worse by demanding `speaker`
be a name from the roster *while also* saying "write 主持 if unsure" — a
contradiction it settled by guessing. Caught in the wild: a NVIDIA call at
120 attributed to KC, when the transcript shows KC saying 「你唔係120蚊咩，我
記得」 — quoting **Eugene's** earlier number back at him. Voice ID later
confirmed KC was not even the speaker in that segment (score 0.35).

Two things that look like fixes and are not:

- **Vocative cues** ("KC你點睇"). Measured across the whole corpus: 6 hits in
  9,793 segments, **0.1%**. Useless as a primary mechanism.
- **Asking the model to try harder.** It has no signal to reason from. More
  instruction produces more confident guessing, not better attribution.

The working approach is speaker *identification*, not diarization: enrol a
voiceprint per host, then match each caption segment against it.

- **Enrolment is free for anyone with a solo video** — every second is
  definitionally them. Check the title first: 【由錢入心】 is an interview and
  「嘉賓」 means a guest is present. A voiceprint averaged over two people
  quietly attributes one's calls to the other.
- **Thresholds are calibrated on this corpus, not from a paper.** Different
  hosts here score ~0.40 against each other — same language, same subject,
  similar recording chain — where unrelated audio scores ~0.1. A threshold
  taken from published benchmarks would be far too low. Measured: 0.55 admits
  0% of a different host, 0.60 costs one point of recall and buys headroom.
- Two gates, both required: absolute threshold *and* margin over the runner-up.
  Near-equal scores mean cross-talk, which is exactly where attribution must
  not be attempted.
- **Voice outranks the model.** Where identification has run, its answer
  replaces whatever name the model produced, and its refusal drops the name
  entirely rather than falling back to the guess. `views.attribution` records
  which mechanism was used — `voice` counts are trustworthy, `guessed` ones
  predate this and are not.
- Audio is downloaded per video and **deleted in a `finally`**. A 2.5-hour show
  is ~570 MB of 16 kHz wav against ~36 KB of transcript, and it is useless once
  embedded.
- Read wavs with stdlib `wave`, not torchaudio: `fetch_audio` always writes
  16 kHz mono 16-bit PCM, and torchaudio moved its top-level `load`/`info` out
  from under us at 2.13.

## Long videos get a thinner summary — prefer the parts

Measured on the same episode of 錢錢錢打到嚟, which the channel posts both as
one 2h31m stream and as four ~35min parts. Near-identical transcript
(35,882 vs 35,595 chars):

| | full 2h31m | as 4 parts |
|---|---|---|
| claims | 28 | 45 |
| figures cited | 136 | 534 |
| verified | 98% | 99% |

Almost 4x the detail at the same accuracy. One summary stretched over 2.5
hours is thin; four summaries each covering ~35 minutes are not. Where a
channel publishes both, filter out the full-length version — RagaFinance's
`title_exclude` does exactly that.

Practical consequence: if a single long video ever has to be summarised
directly, chunking it and summarising each chunk will beat one pass, even
though the whole transcript fits the context window comfortably. Fitting is
not the same as being covered well.

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

.venv/bin/ytdigest scorecard       # per-speaker record, with exclusions shown
.venv/bin/ytdigest voices          # enrolled voiceprints
.venv/bin/ytdigest enroll <name> --video <solo-video-id> [--video ...]
.venv/bin/ytdigest identify <video-id>   # attribute segments; deletes audio after
```

Speaker identification needs the optional extra:
`uv pip install --python .venv/bin/python -e '.[voice]'` (~2 GB, pulls torch).
Without it the pipeline still runs — it just attributes nobody, which is the
safe direction.

## Build order

M0/M1/M3/M4 are done; M2 (ASR) is deliberately unbuilt — see `STATUS.md` for
the decision rule. Do not build stages speculatively ahead of need.
