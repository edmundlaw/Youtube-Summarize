# ASR bake-off results

**Verdict: the kill criterion is passed decisively. Build it.**
**Recommended model: `mlx-community/Qwen3-ASR-1.7B-bf16`, biased with the term list.**

Run 2026-08-11 on an M4 / 16 GB. Design: `2026-08-11-asr-cross-check-design.md`.

## Method

Two windows of `MgN00MCDDRM` (the 2h31m 錢錢錢 episode), chosen because the
stored captions are known to be wrong there:

- **W1** 7060–7130 s — a rapid list of tickers
- **W2** 7470–7545 s — northbound flow figures, the number-critical case

Full `bestaudio` was downloaded once (154 MB webm) and sliced locally to 16 kHz
mono. `--download-sections` was tried first and silently truncated 485 s to 69 s,
so it is not trustworthy for this.

Candidates: `whisper-large-v3-turbo` and `Qwen3-ASR-1.7B-bf16`, each run plain
and biased with a ~60-word term list of tickers and finance vocabulary.
SenseVoice was not reached; the result was already decisive.

## Scorecard

`Y` = correct, `N` = the known-wrong form, `-` = not said in that window.
口語 = count of 嘅咗喺唔㗎咧啲, i.e. whether colloquial Cantonese survived.

| run | 中芯 | 華虹 | 9988 | 淨流入 | 299億 | 紫金 | 981 | 1347 | 信達生物 | 口語 |
|---|---|---|---|---|---|---|---|---|---|---|
| **captions W1** | N | N | N | - | - | N | Y | Y | N | 25 |
| **captions W2** | N | N | - | N | N | - | - | - | - | 17 |
| whisper plain W1 | - | N | - | - | - | N | Y | Y | Y | **0** |
| whisper biased W1 | Y | Y | Y | - | - | N | Y | Y | Y | **0** |
| whisper plain W2 | Y | N | - | Y | Y | - | - | - | - | **0** |
| whisper biased W2 | Y | Y | - | Y | Y | - | - | - | - | **0** |
| qwen plain W1 | N | N | Y | - | - | Y | Y | Y | Y | 34 |
| **qwen biased W1** | N | Y | Y | - | - | Y | Y | Y | Y | 34 |
| qwen plain W2 | N | N | - | Y | Y | - | - | - | - | 30 |
| **qwen biased W2** | Y | Y | - | Y | Y | - | - | - | - | 30 |

Captions score **2 correct out of 11**. Both biased models score 9–10.

## The finding that justifies the whole project

> captions: 中芯國際北水**正留入29億** … 一個就係**接近三百億**
> qwen:     中芯國際北水**淨流入二百九十九億** … 一個就係**接近三百億**
> whisper:  中芯國際北水**淨流入299億** … 一個接近**300億**

**The captions are 10x wrong on a published figure.** They also contradict
themselves 30 seconds apart — 29億 against 三百億 — while both ASR models are
internally consistent at 299億 ≈ 300億, and the surrounding comparison (華虹
56億, 騰訊 150億, 阿里 381億) only makes sense at ~300億.

Two independent models agree against the captions. Run through the repo's own
`find_numbers`, the caption ledger yields `2,900,000,000` where ASR yields
`29,900,000,000`. This figure is in the production database right now, and no
existing check could ever have caught it: the validator compares the summary
against a ledger built from the same wrong transcript.

Captions also dropped an entire provenance sentence — 摩根大通嘅網站 … 四月一號
到六月三十號 — which both models recovered. That is the omission class the
targeted mode is structurally blind to, and it is the argument for a full pass.

## Why Qwen wins despite a lower raw term score

Whisper biased edges it on terms (10Y/1N vs 9Y/1N) and uses a third of the
memory. It loses on three counts that matter more:

1. **It destroys 口語 completely — a score of 0 on every run.** Output is
   Mandarin-phrased Simplified Chinese: 嘅咗喺唔 all gone. `AGENTS.md` forbids
   this because it changes meaning. Disqualifying for the full-transcript role
   (Ron Lau's channel), where ASR output *becomes* what the summariser quotes.
2. **Biasing injects its own errors.** With 紫金礦業 in the prompt, Whisper
   turned 資金流 (capital flow) into 紫金流. Qwen, given the same list, kept
   資金流 correct and still got 紫金 right where Whisper produced 指金.
3. **It degenerates.** The prompted W2 run collapsed into `라` repeated several
   hundred times. Qwen exposes `repetition_penalty`; Whisper's `initial_prompt`
   made the loop more likely, not less.

Qwen's single remaining miss is 中芯 in W1, inside a fast ticker list; it gets it
right in W2 where the name is spoken deliberately.

## Cost

Both models run at **≈4.8x real time** warm. Whisper's first call measured
1.87x, which was model load, not throughput — warm figures are the real ones.

- Targeted mode: 29% of 63 h/month = 18 h audio → **~3.8 h compute/month**
- Full mode: 63 h/month → ~13 h compute/month
- A 2h31m video, full: ~31 minutes

Peak memory: **Qwen 5.2 GB**, Whisper 1.7 GB. On a 16 GB machine already holding
~8 GB this is the real constraint, and it is why ASR must keep its own
subprocess. It also argues for running ASR and voice ID sequentially within the
audio stage, never concurrently.

## Blocker found: the numeral parser corrupts spoken digit strings

Qwen renders tickers the way they are spoken — 七零零, 九九八八, 一三四七, 九八一.
Fed to the existing parser:

| spoken | `cn_to_number` | should be |
|---|---|---|
| 七零零 | `None` | 700 |
| 九九八八 | **8.0** | 9988 |
| 一三四七 | **7.0** | 1347 |
| 九八一 | **1.0** | 981 |

It does not merely miss them — **it returns the last digit as the value, silently**.
On W1 the caption ledger produced 10 figures and Qwen's produced **zero valid
ones**, because every ticker either vanished or resolved to a bogus small integer.

Compound forms are fine: 二百九十九億 → 29,900,000,000 and 五十六億 →
5,600,000,000 both parse correctly. The gap is specifically digit-by-digit
readings, and `一點七倍` (1.7x) also returns `None`.

**This must be fixed before Qwen output reaches the ledger**, or the cross-check
will manufacture disputes and agreements out of noise. Whisper does not have this
problem — it emits Arabic digits — but Whisper is disqualified on 口語.

Whisper's W2 run also parsed `4月1日至6月30日` into the figures 1, 4, 6 and 30.
Date handling needs the same care the repo already applied to clock times.

## Consequences for the design

1. Model is Qwen3-ASR-1.7B-bf16 with `system_prompt` biasing and
   `repetition_penalty` set.
2. **New prerequisite:** extend `numbers.py` to parse digit-string readings, and
   make the failure mode explicit rather than returning the last digit.
3. Term list should be built per video from `instruments.yaml` plus the video's
   own resolved instruments — not a fixed global list. Biasing pulls output
   toward whatever is supplied, so a narrower, more relevant list is safer.
4. `AGENTS.md` names `mlx-community/Qwen3-ASR-1.7B-8bit`; the published MLX
   variant is `-bf16`. Corrected.
5. Full mode earns its cost independently of the cross-check: it is the only way
   to catch what the captions omitted entirely, and the only way Ron's channel
   produces anything at all.
