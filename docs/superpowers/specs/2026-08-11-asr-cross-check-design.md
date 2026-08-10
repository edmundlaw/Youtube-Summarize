# Cross-checking captions with our own ASR

**Status:** design approved 2026-08-11. Bake-off gates everything below it.

## The problem

YouTube's auto-captions mishear this corpus systematically, and nothing in the
pipeline can currently tell. Measured across 5,074,981 characters of stored
normalized transcript (73 videos):

- **中芯 appears 0 times. 中心 appears 288 times.** SMIC is discussed constantly
  on these channels and has never once been transcribed correctly. One published
  digest recommended buying 中心國際, a company that does not exist.
- **華虹 is mangled inconsistently** — 華紅力 in one sentence, 紅能力 minutes
  later, in the same video.
- **Digits are corrupted.** Alibaba's ticker 9988 appears as `九8八`: Chinese and
  Arabic numerals mixed, one digit short.
- **沽 (to sell/short) → 孤 (lonely)**, 349 occurrences against 97 correct.
- **淨流入 (net inflow) → 正留入.**

The validator cannot see any of this. It compares the summary's figures against a
`number_ledger` extracted from *the same transcript*, so a mis-heard number is
mis-heard identically in both and passes as verified. The corpus-wide 97.4%
verified rate (5,438 figures) therefore measures the model's fidelity to the
transcript, **not the transcript's fidelity to the speaker**. This is a blind
spot, not a clean bill of health.

Wrong numbers are the one unacceptable failure for this project. A blind spot
sitting directly over them is the highest-value thing to fix.

## What we are building

A second, independent opinion on every figure, produced locally, compared
against the captions. Where they agree the figure is confirmed for the first
time. Where they disagree it is refused, not resolved.

**Not** a replacement transcript for videos that have captions. **Not** a cloud
service — at 63 h of audio/month the dedicated ASR APIs cost roughly $15–60/mo,
which the user declined; local is also the better technical answer here because
the strongest Cantonese models are open-weight.

## Architecture

One module, `src/ytdigest/asr.py`, one engine, two entry points:

- `transcribe_full(wav) -> list[Segment]` — whole file.
- `transcribe_spans(wav, spans) -> list[Segment]` — listed time ranges only.

Two callers:

| caller | mode | why |
|---|---|---|
| `transcribe` stage | full | a video with no usable captions has no transcript at all — this is Ron Lau's entire channel |
| audio stage (`identify`) | spans | the ledger already knows where every figure is |

The audio stage is the right home for the cross-check because it **already
downloads the wav** for voice identification, already reads 16 kHz slices out of
it with `voice.read_slice`, and already deletes it in a `finally`. Marginal cost
of the cross-check is CPU only — no second download, no second 570 MB.

It also runs after `normalize`, which is where the ledger is built, so figures
and their timestamps exist by then. Ordering is already correct; no stage moves.

The stage name stops describing its contents once it does two unrelated audio
jobs. A rename to `audio` is correct but costs a migration over `videos.status`
for cosmetic benefit, so it is deferred.

Both voice ID and ASR are optional installs. Each degrades independently: no
voice extra means nobody is attributed, no ASR extra means figures are
`unchecked`. Neither failure fails the video.

## Why only 29% of the audio

The ledger holds **6,315 figures across 73 videos — 87 per video**, each with a
timestamp. Merged ±8 s windows around every figure cover:

- **29% of total corpus audio** (14.7 h of 50.6 h)
- median video 27%, best 7%, worst 50%

So the cross-check costs roughly one third of what a full second transcript
would, for the same protection on the thing that matters.

**The gap this leaves, and why the bake-off measures it:** a window can only
exist where the captions already found a number. A figure the captions dropped
entirely produces no ledger entry, therefore no window, therefore no check. The
targeted mode is structurally blind to omissions. Only a full pass finds them.
The bake-off runs both modes on the same video specifically to measure how many
figures a full pass finds that the captions missed. If that number is large,
the design should change to full-transcript cross-checking despite the 3.4x
cost.

## Data model

`number_ledger` gains three columns (one migration):

- `asr_normalized` — what our own ASR heard at that moment, parsed by the
  existing Chinese-numeral parser
- `crosscheck` — `agreed` | `disputed` | `absent` | `unchecked`
- `asr_model` — which model said so, so a later model change is traceable

`confidence` finally carries meaning. It has been `None` for every caption-based
segment since the start, deliberately, because auto-captions supply no score and
defaulting it to 1.0 would have let the validator treat unverified figures as
verified. Cross-source agreement is a real, earned value.

**Comparison is on `normalized`, never `raw_text`.** Both DeepSeek models and
every ASR candidate rewrite 百分之十三 as 13% unprompted; the Chinese-numeral
parser (億/萬/千萬/成/分之) is load-bearing on both sides of the comparison.

### The four states, and why `absent` is separate

- **agreed** — both heard the same normalized value. Confirmed.
- **disputed** — both heard a number, different values. Refused.
- **absent** — ASR heard no number in that window. **Not** a disagreement.
  Silence, cross-talk or a muffled passage is not evidence the caption is wrong,
  and treating it as such would flag a large share of figures on day one.
- **unchecked** — ASR did not run: not installed, timed out, or the machine was
  under pressure.

## Behaviour on disagreement

A disputed figure returns verdict `flagged`, which the validator already refuses
to pass and `publish` already renders with `⚠︎`. No new verdict, no new
rendering path. The reason string carries both readings:

> 字幕係 13%，我哋自己聽係 30%

**Digests will get worse-looking before they get better.** Figures that pass
silently today will start coming back flagged, because for the first time
something can tell they were never checked. That is the feature.

## The bake-off, which gates the build

Nothing above gets built until this passes.

**Sample:** three real videos — one figure-dense episode, one 2.5 h stream (the
worst case for both memory and drift), and one of Ron's, which has no captions
at all and therefore exercises full mode where there is nothing to lean on.

**Candidates:**

- `mlx-community/Qwen3-ASR-1.7B-8bit` — already verified to exist (2026-07-25);
  built for Chinese including dialects and code-switching; accepts a context
  prompt
- Whisper large-v3-turbo via MLX
- SenseVoice-Small — explicit `yue` support, fast, but utterance-level
  timestamps may be too coarse

Each run **twice: plain, and with the ticker list and glossary supplied as a
context prompt.** This is the hypothesis with the highest expected value —
中芯 and 華虹 look like a vocabulary failure, not an acoustic one, and if
biasing fixes them it is by far the cheapest win available.

**Scored on:**

1. The known-wrong list: 中芯, 華虹, 9988, 沽, 淨流入. Primary criterion.
2. Figure agreement against the caption ledger, and on each disagreement, which
   source is actually right — established by listening to the slice.
3. **Figures found by a full pass that the captions missed entirely** — the
   omission gap above.
4. Wall clock per hour of audio, and peak RSS.
5. 口語 fidelity (嘅/咗/喺/唔 preserved, not normalised to written Chinese).
   **Tiebreak only, and only for the full-transcript role** — in cross-check
   mode the ASR prose is never read by anyone, only its numbers are extracted,
   so normalisation there costs nothing. It matters on Ron's channel, where ASR
   output *becomes* the transcript the summariser quotes.

**Kill criterion:** if the best candidate does not beat the captions on
criterion 1, stop and build nothing.

## Machine constraints

63 h of audio/month at current volume (67 videos, 42 h, over 20 days; 38 min
average). Cross-checking 29% is ~18 h of audio/month through ASR.

At 5x real-time that is under 4 h of compute/month and a non-issue. At 1x it
roughly doubles end-to-end time per video on a machine already holding 8 GB of
9 GB with a dozen other services on it. The bake-off measures this before
anything is committed to.

ASR runs in its own subprocess like every other stage — MLX does not reliably
return unified memory to the OS within a process. Sequential only.

**The cross-check must never fail a video.** Missing model, timeout, or memory
pressure leaves figures `unchecked` and the video publishes exactly as it does
today. A safety feature that takes the pipeline down is a worse bug than the one
it fixes.

## Testing

The repo's own warning applies: the stubbed DeepSeek tests could not see any of
the real transport failures. Stubs alone are insufficient here too.

- Unit tests over the comparison logic — `agreed`, `disputed`, `absent`,
  `unchecked` — against a hand-built fixture containing a known 13-vs-30
  disagreement, pinning that it reaches the digest as `⚠︎` carrying both
  readings.
- Unit test that comparison uses `normalized` and not `raw_text`.
- Unit test that an ASR failure leaves the video publishable.
- The models themselves are exercised by the bake-off on real audio, not by
  stubs.

## Explicitly not building

- No cloud tiebreaker on disputed slices (~$1–2/mo) — declined.
- No replacing captions as the primary transcript where captions exist.
- No automatic re-run over the 67 existing videos. Re-checking the back
  catalogue is ~15 h of compute and would quantify how much of the stored record
  is wrong; worth doing, but as a separate decision once this works.
- No stage rename.
