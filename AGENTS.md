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
- **There are no dated model IDs.** DeepSeek ships new builds behind the same
  two names — probing `deepseek-v4-0731`, `-flash-0731`, `-pro-0731` and
  friends all return *"The supported API model names are deepseek-v4-pro or
  deepseek-v4-flash"*. A release announcement therefore means the pipeline is
  already on the new build, with no way to pin the old one. Re-test after one.
- **Re-tested 2026-08-02 on the 0731 build. Flash stays, for a stronger
  reason than the glossary.** Same video, same voice labels, both models:

  | | flash | pro |
  |---|---|---|
  | speaker attributions matching voice ID | 8/9 | 4/8 |
  | mismatches that **invent** a name | **0** | **4** |
  | theses / actionable | 3 / 1 | 4 / 3 |
  | wall clock | 461 s | 369 s |

  Pro is faster and produces more claims. It also named 顧芷筠 (Debby) and
  Eugene 羅尚沛 at four timestamps the transcript had labelled 主持 — neither
  has a voiceprint, so nothing could ever confirm them. Flash's single
  mismatch was the opposite: it wrote 主持 where voice knew it was KC.

  That asymmetry is the whole decision. Flash under-claims and costs a data
  point; pro fabricates and corrupts a person's record. `store_views` overrides
  with voice where identification has run, but the digest prose still carries
  the invented name, and on a video with no identification nothing overrides it
  at all.

  The glossary axis was **not** re-tested: the sample video contains exactly
  one protected term (`yoy`) and neither model kept it, which is too thin to
  conclude anything. Re-run on a vocabulary-rich episode before revisiting.

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

## DeepSeek transport: stream, and read finish_reason

All of this was found by running real videos, not by tests. The unit tests
stub the API and could not see any of it.

- **Requests must stream.** Buffered, a summarise call spends minutes with no
  bytes on the socket while the model reasons, and DeepSeek cuts it: *"peer
  closed connection without sending complete message body"* on all three
  attempts. Streaming removes it. `usage` arrives in a final frame carrying no
  `choices`, so accumulate `finish_reason` and `usage` separately rather than
  reading the last chunk.
- **Reasoning tokens dominate `max_tokens`.** A trivial 7-character answer
  measured 221 reasoning tokens of 227. A budget sized for the answer returns a
  fragment. One 37-minute options show spent 15,795 of 16,000 thinking and
  emitted 590 characters, so no fixed value survives every video —
  `_Truncated` grows it 1.6x per attempt to a 40k ceiling.
- **`finish_reason` decides the remedy, not whether content survived.** Reason
  `length` is a truncation whether a fragment escaped or nothing did; both need
  a bigger budget. Only an empty completion with `stop` is transient and worth
  an identical retry. Splitting those two apart meant the budget escalation
  never fired on the video it was written for.
- **Never conclude "raise max_tokens" from an empty completion alone.** Doing
  so once took the budget to 16000, which made things worse: the longer
  generation was the one being cut. Check `finish_reason` first.
- A truncated body reaches the JSON parser as *"unterminated string"*, which
  points nowhere near the cause. Truncation is raised where it happens.

## Regenerating summaries

`ytdigest resummarize [ids] [--identified] [--stale-prompt]` re-runs generation
only; `retry` re-runs every stage including fetch. Telegram is off by default —
correcting months of stored summaries must not replay months of digests.

**Views are replaced per video, not merged.** A regenerated summary supersedes
the one before it, but the dedupe key includes `speaker`, so the moment
attribution changes — a guessed name becoming refused, which is the entire
point of voice identification — the old row survives beside the new one and the
scorecard counts both. Observed as 49 stale rows across 5 regenerated videos.

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

## Who is on the episode is read off the title, and titles lie

Two independent filters, and they are not interchangeable. `title_include` is
per channel and runs at **discovery**, so a video it rejects is never stored.
`in_focus` runs at **claim time** on each pass, because an episode's 第一節 /
第二節 parts carry no host names and a part discovered before its parent cannot
be judged yet.

Use `title_include` when a channel names its presenter and rotates them:
JK爸爸的投資頻道 runs 午後開股 daily, and only 6 of 16 recent episodes are JK Sir
— the rest are Gary Sir, Ringo, Jason Sir. Use nothing when a channel is one
person: 全職炒家's weekly 【熱點先機】 names nobody at all and is always Ron, so
a name-based include would drop his main output. Do not reach for `in_focus` to
do either job; it keeps anything it cannot judge, on purpose.

Title formats that have each broken the parser, all found on real videos:

- **`主持 Wendy`** — space, no colon. Requiring the colon meant "no declaration",
  which fell through to the hashtag branch and answered `港股, 美股`.
- **`#恆指 #倍升股 #牛市`** — the hashtag fallback exists for 1號月台, which tags
  its guest `#羅家聰`. On a channel that tags topics it declared three market
  indices to be the speakers. A tag counts only if it resolves to the roster.
- **`｜RON LAU｜主持 Wendy`** — the declaration names the moderator, not the
  analyst everyone watches for. That list is also the whitelist `parse_views`
  filters attributions against, so a moderator-only roster silently **discards
  every view Ron makes on his own channel**. A declaration is widened by anyone
  else the title names; roster names alone never create one, because
  `JK Sir｜Jason Sir｜Car` declares nobody and "the sole host is JK Sir" would
  hand him the other two's calls.
- **`|| 羅家聰||`** — no keyword at all, so `hosts_from_title` returns
  `哈富證券||26-07-22`. This is why `in_focus` checks focus aliases against the
  raw title before it looks at any parsed host list.

Aliases are matched as lowercased substrings of the whole title, so short ones
are dangerous — bare `ron` matches Micron, which these channels discuss by name.
`focus_aliases` already drops anything under two characters; keep the rest long
enough to mean only the person.

RSS is the only listing that sees everything. `list_uploads` reads the uploads
tab, which **omits live streams** — both channels' 直播 and 午後開股 are absent
from it and present in the feed. It also carries no duration, so a channel's
`min_duration_s` cannot reject at discovery: 全職炒家's 50-second 紅綠燈AI模型
product ads are stored first and rejected at fetch.

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
