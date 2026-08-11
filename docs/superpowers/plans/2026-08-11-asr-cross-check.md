# ASR Cross-Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every figure in the ledger a second, independent reading from local ASR, so a mis-heard number is refused instead of published as verified.

**Architecture:** The `number_ledger` already knows the timestamp of all 87 figures in a typical video. The audio stage already downloads a wav for voice ID and deletes it in a `finally`. This plan slices ±8 s around each figure out of that same wav, transcribes those spans with Qwen3-ASR, re-parses them with the existing number parser, and records `agreed` / `disputed` / `absent` / `unchecked` on each ledger row. A disputed figure returns the existing `flagged` verdict, which the validator already refuses and the digest already renders with `⚠︎`.

**Tech Stack:** Python 3.12, MLX via `mlx-audio`, `mlx-community/Qwen3-ASR-1.7B-bf16`, SQLite, pytest.

## Global Constraints

- **Never fail a video.** Any ASR failure — missing model, timeout, memory pressure — leaves rows `unchecked` and the video publishes exactly as it does today.
- **Compare on `normalized`, never `raw_text`.** Both DeepSeek and every ASR model rewrite 百分之十三 as 13% unprompted.
- **Model id is `mlx-community/Qwen3-ASR-1.7B-bf16`.** The `-8bit` id in `config/config.toml` does not exist and must be corrected.
- **Colloquial Cantonese (嘅/咗/喺/唔) is preserved, never normalised.** This is why Whisper was rejected.
- **Sequential only.** ASR peaks at 5.2 GB on a 16 GB machine already holding ~8 GB. Never run ASR and voice ID concurrently.
- **Schema changes use `db.split_statements()`, never `executescript()`** — it issues an implicit COMMIT and breaks out of the transaction.
- Filenames are always `<video_id>`.
- Every task ends with `.venv/bin/python -m pytest -q` green and `.venv/bin/python -m pyflakes src/ytdigest tests` clean.

## Out of scope for this plan

**Full-transcript mode for Ron Lau's channel is a separate plan.** The bake-off found that Qwen3-ASR returns a *single* segment spanning its whole input — `{'start': 0.0, 'end': 75.0}` for a 75 s clip — with no internal timestamps. Cross-check does not care, because we choose the spans and therefore already know their times. Full mode does care: views need `start_s` and voice ID needs per-segment boundaries, so it first requires VAD chunking (silero-vad, already listed in the `asr` extra, with `chunk_target_s` / `min_silence_ms` already in `config.toml`). Attempting both in one plan would couple an approved, well-measured feature to an unbuilt one.

---

### Task 1: Stop the numeral parser returning the last digit of a spoken digit string

`cn_to_number` walks the string assigning each digit to `current`, so a run of bare digits with no magnitude word ends up as whichever digit came last. `九九八八` — Alibaba's ticker, as Qwen speaks it — returns `8.0`. It does not fail; it lies. This is live in production today, independent of anything else in this plan.

**Files:**
- Modify: `src/ytdigest/numbers.py:58-101`
- Test: `tests/test_numbers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cn_to_number(text: str) -> float | None` now returns the concatenated value for runs of 3+ bare digits, `None` for runs of exactly 2, and is unchanged for compositional numerals.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_numbers.py`:

```python
def test_spoken_digit_strings_are_not_read_as_their_last_digit():
    """九九八八 is Alibaba's ticker 9988, spoken digit by digit. The
    compositional parser overwrote its accumulator on each digit and returned
    8.0 -- silently wrong, which is worse than refusing. Qwen3-ASR renders every
    ticker this way, so without this the cross-check compares noise."""
    from ytdigest.numbers import cn_to_number

    assert cn_to_number("九九八八") == 9988
    assert cn_to_number("七零零") == 700
    assert cn_to_number("一三四七") == 1347
    assert cn_to_number("九八一") == 981


def test_two_digit_runs_are_refused_as_ambiguous():
    """兩三 is "two or three" -- an approximation, not 23. Guessing here would
    invent a figure, which is the one failure this project does not accept."""
    from ytdigest.numbers import cn_to_number

    assert cn_to_number("兩三") is None
    assert cn_to_number("三四") is None
    assert cn_to_number("五六") is None


def test_compositional_numerals_are_unaffected():
    """The forms that already worked must not regress: these carry every real
    figure in the corpus."""
    from ytdigest.numbers import cn_to_number

    assert cn_to_number("二百九十九") == 299
    assert cn_to_number("十三") == 13
    assert cn_to_number("三萬五千") == 35000
    assert cn_to_number("五") == 5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_numbers.py -k "digit_string or two_digit or compositional" -v`
Expected: FAIL — `assert 8.0 == 9988`.

- [ ] **Step 3: Implement**

In `src/ytdigest/numbers.py`, add this function immediately above `cn_to_number`:

```python
def _digit_string(text: str) -> float | None:
    """七零零 -> 700. Digits read out one at a time, which is how tickers and
    some index levels are spoken, and how Qwen3-ASR renders both.

    Runs of exactly two are refused: 兩三 / 三四 / 五六 are "two or three", an
    approximation, and returning 23 would fabricate a figure. Three is the
    shortest length at which a digit string is unambiguous in this corpus.
    """
    if len(text) < 3:
        return None
    return float("".join(str(_DIGITS[c]) for c in text))
```

Then in `cn_to_number`, immediately after the existing character guard
(`if not text or any(c not in _CN_NUM for c in text): return None`), insert:

```python
    # A run of bare digits carries no magnitude word, so it is not
    # compositional and the loop below cannot parse it -- that loop assigns
    # each digit to `current`, so 九九八八 would fall out the bottom as 8.0.
    # Returning the wrong number silently is the failure this project exists to
    # prevent, so digit strings are handled here or refused.
    if all(c in _DIGITS for c in text):
        return _digit_string(text)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_numbers.py -v`
Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 5: Run the full suite — this function has many callers**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m pyflakes src/ytdigest tests`
Expected: 162 passed, pyflakes silent.

- [ ] **Step 6: Commit**

```bash
git add src/ytdigest/numbers.py tests/test_numbers.py
git commit -m "Parse spoken digit strings instead of returning their last digit

cn_to_number walks the string assigning each digit to an accumulator, so a
run of bare digits with no magnitude word came out as whichever digit was
last: 九九八八, Alibaba's ticker, returned 8.0. It did not fail, it lied.

Runs of three or more now parse whole. Runs of two are refused -- 兩三 is
two or three, an approximation, and 23 would be a fabrication."
```

---

### Task 2: Let the ledger find spoken digit strings at all

Task 1 makes `cn_to_number` parse `七零零`. The ledger still will not see it: `_PATTERNS` deliberately excludes bare Chinese numerals, because 一 and 十 occur constantly in prose. A three-digit run is different, and without this the caption ledger (Arabic `700`) and the ASR ledger (spoken `七零零`) have nothing in common to compare.

**Files:**
- Modify: `src/ytdigest/numbers.py:192-234`
- Test: `tests/test_numbers.py`

**Interfaces:**
- Consumes: `_digit_string` and the `cn_to_number` behaviour from Task 1.
- Produces: `find_numbers` now emits `Found(unit=COUNT)` for runs of 3+ spoken digits, and `Found(unit=YEAR)` for spoken 4-digit years.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_numbers.py`:

```python
def test_spoken_tickers_enter_the_ledger():
    """YouTube's captions write tickers as Arabic digits and Qwen speaks them.
    Unless both reach the ledger there is nothing to cross-check."""
    from ytdigest.numbers import find_numbers

    got = {f.raw: f.value for f in find_numbers("譬如七零零啦，九九八八啦，一三四七啦")}
    assert got["七零零"] == 700
    assert got["九九八八"] == 9988
    assert got["一三四七"] == 1347


def test_spoken_years_are_years_not_quantities():
    """二零二五年 must classify as YEAR like its Arabic twin, or is_financially
    _meaningful lets a calendar year into the ledger as a figure."""
    from ytdigest.numbers import YEAR, find_numbers

    got = [f for f in find_numbers("到二零二五年為止") if f.unit == YEAR]
    assert got and got[0].value == 2025


def test_ordinary_prose_still_yields_no_bare_chinese_numbers():
    """一 and 十 are everywhere in speech. Only runs of three or more count."""
    from ytdigest.numbers import find_numbers

    assert find_numbers("我一於唔買，十分之危險") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_numbers.py -k "spoken_tickers or spoken_years or ordinary_prose" -v`
Expected: FAIL with `KeyError: '七零零'`.

- [ ] **Step 3: Implement**

In `src/ytdigest/numbers.py`, inside the `_PATTERNS` list, replace the single
existing YEAR line with these two entries, and add the digit-string entry
directly *above* the final bare-Arabic entry:

```python
    # years: 2022年 / 二零二二年
    (YEAR, r"(?:19|20)\d{2}\s*年"),
    (YEAR, rf"[{_CN_D}]{{4}}\s*年"),
```

```python
    # Spoken digit strings: 七零零, 九九八八, 一三四七. Qwen3-ASR renders tickers
    # and some index levels this way where the captions render Arabic digits --
    # without this the two ledgers share nothing and every figure cross-checks
    # as `absent`. Three characters minimum, and the lookarounds stop this
    # stealing the tail of a compositional numeral such as 二百九十九.
    (COUNT, rf"(?<![{_CN_NUM}])[{_CN_D}]{{3,}}(?![{_CN_NUM}])"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_numbers.py -v`
Expected: PASS.

- [ ] **Step 5: Verify against real stored transcripts, not just fixtures**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from ytdigest.config import load_config
from ytdigest import db as D
from ytdigest.numbers import find_numbers, is_financially_meaningful

cfg = load_config(); conn = D.connect(cfg.db_path)
before = conn.execute("SELECT COUNT(*) FROM number_ledger").fetchone()[0]
total = 0
for r in conn.execute("SELECT path FROM artifacts WHERE kind='normalized' LIMIT 20"):
    d = json.loads(Path(r["path"]).read_text("utf-8"))
    for s in d["segments"]:
        total += sum(1 for f in find_numbers(s["text"]) if is_financially_meaningful(f.unit))
print(f"stored ledger rows: {before};  re-extracted over 20 videos: {total}")
PY
```

Expected: the re-extracted count is within a few percent of the stored rows for
those videos. A large jump means the new pattern is firing on prose — stop and
tighten it before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/ytdigest/numbers.py tests/test_numbers.py
git commit -m "Let the ledger see spoken digit strings and spoken years

Bare Chinese numerals are excluded from _PATTERNS because 一 and 十 are
everywhere in speech, but a run of three or more digits is unambiguous and
is how Qwen speaks every ticker. Without it the caption ledger holds 700 and
the ASR ledger holds nothing, so every figure would cross-check as absent."
```

---

### Task 3: The Qwen3-ASR engine

The `ASREngine` protocol and `runner._load_asr` already exist and already take a `chunks: list[SpeechRegion]` argument — that argument *is* spans mode. Nothing new is needed in the seam.

**Files:**
- Create: `src/ytdigest/asr/__init__.py`
- Create: `src/ytdigest/asr/qwen3.py`
- Modify: `config/config.toml:32-40`
- Modify: `pyproject.toml:19-27`
- Test: `tests/test_asr.py`

**Interfaces:**
- Consumes: `interfaces.Segment`, `interfaces.SpeechRegion`, `voice.read_slice(wav_path, start_s, duration_s) -> np.ndarray`.
- Produces: `Qwen3ASRMLX(cfg)` with `.id: str`, `.params_hash() -> str`, and
  `.transcribe(audio_path: Path, chunks: list[SpeechRegion], lang_hint: str | None = None, context: str | None = None) -> list[Segment]`.
  One `Segment` per chunk, carrying that chunk's real start/end. Called with `chunks=[]` it raises `NotImplementedError` — full mode needs VAD and is a separate plan.

- [ ] **Step 1: Write the failing test**

Create `tests/test_asr.py`:

```python
"""The engine is stubbed here. The model itself was measured by the bake-off on
real audio -- see docs/superpowers/specs/2026-08-11-asr-bake-off-results.md.
The repo already learned that stubbed tests cannot see real model behaviour, so
these cover only the wiring: span arithmetic, and that failures stay contained.
"""

import numpy as np
import pytest

from ytdigest.interfaces import SpeechRegion


class _FakeOut:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, audio, **kw):
        self.calls.append((audio, kw))
        return _FakeOut(f"heard{len(self.calls)}")


def _engine(monkeypatch, model=None):
    from ytdigest.asr.qwen3 import Qwen3ASRMLX

    cfg = type("C", (), {"get": lambda self, s, k, d=None: d})()
    eng = Qwen3ASRMLX(cfg)
    eng._model = model or _FakeModel()
    monkeypatch.setattr("ytdigest.voice.read_slice",
                        lambda *a, **k: np.zeros(16000, dtype=np.float32))
    return eng


def test_each_span_becomes_one_segment_at_its_real_timestamp(monkeypatch, tmp_path):
    """The span's own start/end must survive. Qwen returns a single segment
    spanning whatever it is given and reports it as 0.0-to-duration, so if the
    caller does not carry the real offsets every figure lands at t=0."""
    eng = _engine(monkeypatch)
    spans = [SpeechRegion(100.0, 116.0), SpeechRegion(250.5, 266.5)]
    segs = eng.transcribe(tmp_path / "a.wav", spans)

    assert [(s.start, s.end) for s in segs] == [(100.0, 116.0), (250.5, 266.5)]
    assert [s.text for s in segs] == ["heard1", "heard2"]


def test_confidence_stays_none(monkeypatch, tmp_path):
    """Qwen reports no score. Defaulting it to 1.0 would let the validator treat
    an unverified figure as verified -- the same trap auto-captions set."""
    eng = _engine(monkeypatch)
    segs = eng.transcribe(tmp_path / "a.wav", [SpeechRegion(0.0, 8.0)])
    assert segs[0].confidence is None


def test_a_failing_span_does_not_lose_the_others(monkeypatch, tmp_path):
    """One bad span must not cost the whole video its cross-check."""
    class _Flaky(_FakeModel):
        def generate(self, audio, **kw):
            self.calls.append(audio)
            if len(self.calls) == 1:
                raise RuntimeError("mlx blew up")
            return _FakeOut("ok")

    eng = _engine(monkeypatch, _Flaky())
    segs = eng.transcribe(tmp_path / "a.wav",
                          [SpeechRegion(0.0, 8.0), SpeechRegion(20.0, 28.0)])
    assert len(segs) == 1 and segs[0].start == 20.0


def test_full_file_mode_is_refused_rather_than_faked(monkeypatch, tmp_path):
    """Qwen returns one untimed segment for its whole input, so a full pass
    needs VAD chunking first. Returning a single 2.5-hour segment would put
    every view at t=0 and silently break voice ID."""
    eng = _engine(monkeypatch)
    with pytest.raises(NotImplementedError):
        eng.transcribe(tmp_path / "a.wav", [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_asr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ytdigest.asr'`.

- [ ] **Step 3: Create the package**

Create `src/ytdigest/asr/__init__.py`:

```python
"""ASR engines. Optional: `uv pip install -e '.[asr]'`.

Kept a package rather than a module because `runner._load_asr` already imports
`ytdigest.asr.qwen3` by name, and a second engine would sit beside it.
"""
```

- [ ] **Step 4: Write the engine**

Create `src/ytdigest/asr/qwen3.py`:

```python
"""Qwen3-ASR via mlx-audio.

Chosen over Whisper by measurement, not preference. On two windows of a real
episode Whisper scored zero on 口語 in every single run -- it renders 嘅咗喺唔
into Mandarin-phrased Simplified Chinese, which AGENTS.md forbids because it
changes meaning. It also turned 資金流 into 紫金流 purely because 紫金礦業 was in
the bias list, and one run degenerated into several hundred repetitions of one
character. Qwen keeps the Cantonese, gets 中芯國際 and 淨流入 right where the
captions never have, and exposes `repetition_penalty`.

Full details: docs/superpowers/specs/2026-08-11-asr-bake-off-results.md
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..interfaces import Segment, SpeechRegion
from ..logging import get_logger

MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-bf16"
LANG = "yue"
#: Qwen degenerates into repetition without this. Whisper had no equivalent and
#: produced several hundred repetitions of one character on a prompted run.
REPETITION_PENALTY = 1.1

log = get_logger()


class Qwen3ASRMLX:
    id = "qwen3-asr-1.7b-bf16"

    def __init__(self, cfg):
        self._model_id = cfg.get("asr", "model_id", MODEL_ID) or MODEL_ID
        self._penalty = float(cfg.get("asr", "repetition_penalty", REPETITION_PENALTY))
        self._model = None

    def params_hash(self) -> str:
        seed = f"{self._model_id}|{self._penalty}|{LANG}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    def _load(self):
        if self._model is None:
            from mlx_audio.stt.generate import load_model

            self._model = load_model(self._model_id)
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        chunks: list[SpeechRegion],
        lang_hint: str | None = None,
        context: str | None = None,
    ) -> list[Segment]:
        """Transcribe the given spans. One Segment per span, at its real time.

        `context` is Qwen's native biasing input. It is powerful and therefore
        dangerous: given a global term list Whisper turned 資金流 into 紫金流.
        Callers should pass a list scoped to this video, not everything known.
        """
        if not chunks:
            raise NotImplementedError(
                "full-file transcription needs VAD chunking first: Qwen returns "
                "one untimed segment for its whole input, so every figure would "
                "land at t=0. See the ASR cross-check plan, 'Out of scope'."
            )

        from ..voice import read_slice

        model = self._load()
        lang = lang_hint or LANG
        out: list[Segment] = []
        for index, span in enumerate(chunks):
            try:
                audio = read_slice(Path(audio_path), span.start, span.duration)
                said = model.generate(
                    audio, language=lang, system_prompt=context,
                    repetition_penalty=self._penalty,
                )
            except Exception as exc:            # noqa: BLE001 - one span, not the video
                log.warning("asr.span_failed", start_s=span.start,
                            error=str(exc)[:200])
                continue
            text = (said.text or "").strip()
            if not text:
                continue
            out.append(Segment(
                id=index, start=span.start, end=span.end,
                text=text, lang=lang, confidence=None,
            ))
        return out
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_asr.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Correct the config and the extra**

In `config/config.toml`, replace the `model_id` line and add the penalty:

```toml
model_id  = "mlx-community/Qwen3-ASR-1.7B-bf16"   # the -8bit id does not exist
repetition_penalty = 1.1                          # Qwen degenerates without it
```

In `pyproject.toml`, replace the `asr` extra's dependency list with:

```toml
asr = [
    "mlx>=0.21",
    "mlx-audio>=0.2",     # provides Qwen3-ASR; mlx-lm alone cannot load it
    "numpy>=1.26",
    "soundfile>=0.12",
]
```

`onnxruntime` (silero-vad) is dropped: it is only needed for full-file mode,
which is out of scope here, and it should not be installed for a feature that
does not use it.

- [ ] **Step 7: Verify the engine loads for real**

Run: `uv pip install --python .venv/bin/python -e '.[asr]' && .venv/bin/python -c "
from ytdigest.config import load_config
from ytdigest.asr.qwen3 import Qwen3ASRMLX
e = Qwen3ASRMLX(load_config()); print(e.id, e.params_hash()); e._load(); print('model loaded')"`

Expected: prints the id, a 16-char hash, then `model loaded`. This downloads
~3.4 GB once.

- [ ] **Step 8: Commit**

```bash
git add src/ytdigest/asr tests/test_asr.py config/config.toml pyproject.toml
git commit -m "Add the Qwen3-ASR engine, spans mode only

Chosen by measurement: Whisper scored zero on 口語 on every run, rendering
嘅咗喺唔 into Mandarin-phrased Simplified Chinese, which changes meaning.

Full-file mode raises NotImplementedError rather than faking it. Qwen returns
a single untimed segment for whatever it is given, so a whole-video pass
would put every figure at t=0 and break voice ID silently. That needs VAD
chunking and is a separate plan."
```

---

### Task 4: Record the second reading on the ledger

**Files:**
- Create: `migrations/008_ledger_crosscheck.sql`
- Test: `tests/test_crosscheck.py`

**Interfaces:**
- Produces: `number_ledger` gains `asr_normalized TEXT`, `crosscheck TEXT`, `asr_model TEXT`. Existing rows read as `crosscheck IS NULL`, meaning never checked.

- [ ] **Step 1: Write the failing test**

Create `tests/test_crosscheck.py`:

```python
import pathlib
import tempfile

import pytest

from ytdigest import db as D


@pytest.fixture
def conn():
    path = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    return D.open_db(path, pathlib.Path("migrations"))


def test_ledger_carries_the_second_reading(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(number_ledger)")}
    assert {"asr_normalized", "crosscheck", "asr_model"} <= cols


def test_existing_rows_default_to_never_checked(conn):
    """A null crosscheck must mean 'nothing has looked at this', not 'agreed'.
    Every row already in production is in this state."""
    conn.execute(
        "INSERT INTO videos (id, channel_id, title, discovered_at, status) "
        "VALUES ('v', 'c', 't', '2026-01-01', 'new')")
    conn.execute(
        "INSERT INTO number_ledger (video_id, raw_text, normalized, unit, start_s) "
        "VALUES ('v', '13%', '13', 'pct', 10.0)")
    row = conn.execute("SELECT crosscheck FROM number_ledger").fetchone()
    assert row["crosscheck"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_crosscheck.py -v`
Expected: FAIL — the three columns are absent.

- [ ] **Step 3: Write the migration**

Create `migrations/008_ledger_crosscheck.sql`:

```sql
-- A second, independent reading of each figure, from our own ASR.
--
-- The validator compares a summary against a ledger built from the same
-- transcript, so a mis-heard number is mis-heard identically on both sides and
-- passes. Measured on MgN00MCDDRM: the captions say 中芯 北水淨流入 29億 and
-- contradict themselves thirty seconds later with 接近三百億; two ASR models
-- independently hear 299億. A 10x error, invisible to every existing check.
--
-- NULL means never checked, which is what every existing row is. It must never
-- be read as agreement.
ALTER TABLE number_ledger ADD COLUMN asr_normalized TEXT;
ALTER TABLE number_ledger ADD COLUMN crosscheck TEXT;
ALTER TABLE number_ledger ADD COLUMN asr_model TEXT;
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_crosscheck.py -v`
Expected: PASS.

- [ ] **Step 5: Apply to the live database and confirm**

Run: `.venv/bin/ytdigest doctor 2>&1 | tail -20`
Expected: no migration errors; `number_ledger` reports the new columns.

- [ ] **Step 6: Commit**

```bash
git add migrations/008_ledger_crosscheck.sql tests/test_crosscheck.py
git commit -m "Record a second reading of each figure on the ledger

NULL means never checked -- what every existing row is -- and must never be
read as agreement."
```

---

### Task 5: The comparison itself

Pure functions, no I/O, no model. This is where the four states are decided, and it is the part most worth getting exactly right.

**Files:**
- Create: `src/ytdigest/crosscheck.py`
- Test: `tests/test_crosscheck.py`

**Interfaces:**
- Consumes: `numbers.find_numbers`, `numbers.is_financially_meaningful`.
- Produces:
  - `AGREED = "agreed"`, `DISPUTED = "disputed"`, `ABSENT = "absent"`, `UNCHECKED = "unchecked"`
  - `WINDOW_S = 8.0`
  - `spans_for(starts: list[float], duration_s: float, window_s: float = WINDOW_S) -> list[tuple[float, float]]` — merged, clamped
  - `values_in(text: str) -> list[float]`
  - `compare(caption_value: float | None, asr_values: list[float]) -> tuple[str, float | None]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crosscheck.py`:

```python
def test_spans_merge_and_clamp():
    """Figures cluster. Overlapping windows must merge or the same seconds get
    transcribed several times -- measured at 29% of audio when merged."""
    from ytdigest.crosscheck import spans_for

    assert spans_for([100.0, 104.0, 500.0], duration_s=600.0, window_s=8.0) == [
        (92.0, 112.0), (492.0, 508.0)]


def test_spans_never_run_past_either_end():
    from ytdigest.crosscheck import spans_for

    assert spans_for([3.0], duration_s=10.0, window_s=8.0) == [(0.0, 10.0)]


def test_agreement_needs_the_same_value():
    from ytdigest.crosscheck import AGREED, DISPUTED, compare

    assert compare(13.0, [13.0]) == (AGREED, 13.0)
    assert compare(13.0, [30.0, 56.0])[0] == DISPUTED


def test_the_reported_rival_is_the_nearest_one():
    """The digest shows both readings, so it should show the closest competing
    one rather than an unrelated figure from elsewhere in the window."""
    from ytdigest.crosscheck import DISPUTED, compare

    assert compare(13.0, [900.0, 30.0]) == (DISPUTED, 30.0)


def test_hearing_no_number_is_absent_not_disputed():
    """Silence, cross-talk or a muffled passage is not evidence the caption is
    wrong. Treating it as disagreement would flag a large share of figures on
    day one and train the reader to ignore the marker."""
    from ytdigest.crosscheck import ABSENT, compare

    assert compare(13.0, []) == (ABSENT, None)


def test_a_caption_figure_that_never_parsed_cannot_be_judged():
    from ytdigest.crosscheck import UNCHECKED, compare

    assert compare(None, [13.0]) == (UNCHECKED, None)


def test_rounding_noise_is_not_a_dispute():
    """29.9億 written 2990000000.0 against 2990000000 is the same number."""
    from ytdigest.crosscheck import AGREED, compare

    assert compare(2_990_000_000.0, [2_990_000_000.0000001])[0] == AGREED


def test_the_real_disagreement_this_was_built_for():
    """MgN00MCDDRM @7483s. Captions 29億, both ASR models 299億, and the
    surrounding figures only make sense at ~300億."""
    from ytdigest.crosscheck import DISPUTED, compare, values_in

    heard = values_in("中芯國際北水淨流入二百九十九億。華虹宏力淨流入五十六億。")
    assert 29_900_000_000 in heard
    assert compare(2_900_000_000, heard) == (DISPUTED, 29_900_000_000)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_crosscheck.py -v`
Expected: FAIL — `No module named 'ytdigest.crosscheck'`.

- [ ] **Step 3: Implement**

Create `src/ytdigest/crosscheck.py`:

```python
"""Compare the ledger's figures against a second reading from our own ASR.

The validator cannot catch a mis-heard number: it checks a summary against a
ledger built from the same transcript, so both sides are wrong together and the
figure passes. Measured across the corpus, 中芯 appears 0 times and 中心 288 --
and on MgN00MCDDRM the captions put SMIC's northbound inflow at 29億 while two
independent ASR models hear 299億.

Nothing here resolves a disagreement. A disputed figure is refused, never
overwritten: ASR is not ground truth either, and replacing one unverified
number with another would assert something no one checked.
"""

from __future__ import annotations

from .numbers import find_numbers, is_financially_meaningful

AGREED = "agreed"
DISPUTED = "disputed"
ABSENT = "absent"
UNCHECKED = "unchecked"

#: Seconds either side of a figure. Merged windows of this size cover 29% of
#: corpus audio, against 100% for a full second transcript.
WINDOW_S = 8.0

#: Relative tolerance. Wide enough to absorb float formatting, far too tight to
#: let 29億 pass as 299億.
_TOLERANCE = 1e-6


def spans_for(starts: list[float], duration_s: float,
              window_s: float = WINDOW_S) -> list[tuple[float, float]]:
    """Merged, clamped windows around each figure.

    Figures cluster -- an analyst reads six numbers off one chart -- so
    unmerged windows would transcribe the same seconds repeatedly.
    """
    spans: list[tuple[float, float]] = []
    for start in sorted(s for s in starts if s is not None):
        lo = max(0.0, start - window_s)
        hi = min(duration_s, start + window_s)
        if hi <= lo:
            continue
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return spans


def values_in(text: str) -> list[float]:
    """Every financially meaningful figure ASR heard in one span."""
    return [
        f.value for f in find_numbers(text)
        if f.value is not None and is_financially_meaningful(f.unit)
    ]


def _same(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= _TOLERANCE


def compare(caption_value: float | None,
            asr_values: list[float]) -> tuple[str, float | None]:
    """One figure against everything ASR heard near it.

    Returns the verdict and, when disputed, the nearest rival reading -- the
    digest shows both, and the closest one is the one worth showing.
    """
    if caption_value is None:
        return UNCHECKED, None          # nothing to compare against
    if not asr_values:
        return ABSENT, None             # silence is not disagreement
    if any(_same(caption_value, v) for v in asr_values):
        return AGREED, caption_value
    rival = min(asr_values, key=lambda v: abs(v - caption_value))
    return DISPUTED, rival
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_crosscheck.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ytdigest/crosscheck.py tests/test_crosscheck.py
git commit -m "Compare each ledger figure against a second reading

Four states, and the distinction that matters is absent vs disputed: ASR
hearing no number is silence or cross-talk, not evidence the caption is
wrong. Conflating them would flag a large share of figures on day one and
teach the reader to ignore the marker.

Nothing here resolves a disagreement. ASR is not ground truth either."
```

---

### Task 6: Run it, inside the stage that already has the audio

**Files:**
- Modify: `src/ytdigest/runner.py:214-235`
- Test: `tests/test_crosscheck.py`

**Interfaces:**
- Consumes: `crosscheck.spans_for/values_in/compare`, `asr.qwen3.Qwen3ASRMLX`, `voice.audio_for`, `views.load_instruments`.
- Produces: `runner._crosscheck_figures(cfg, conn, video, wav) -> dict[str, int]` returning counts per verdict; `runner._bias_terms(conn, video_id) -> str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crosscheck.py`:

```python
def test_a_dead_asr_leaves_rows_unchecked_and_never_raises(conn, monkeypatch, tmp_path):
    """A safety feature that takes the pipeline down is a worse bug than the one
    it fixes. Every failure path must end with the video still publishable.

    `wav` must be a real path, not None: None short-circuits before the engine is
    ever loaded, so the test would pass without exercising the failure at all.
    """
    from ytdigest import runner

    conn.execute("INSERT INTO videos (id, channel_id, title, discovered_at, status) "
                 "VALUES ('v','c','t','2026-01-01','normalized')")
    conn.execute("INSERT INTO number_ledger (video_id, raw_text, normalized, unit, start_s) "
                 "VALUES ('v','29億','2900000000','hkd',100.0)")
    conn.commit()

    loaded = []

    def _boom(cfg):
        loaded.append(True)
        raise RuntimeError("no mlx on this machine")

    monkeypatch.setattr(runner, "_load_asr", _boom)
    cfg = type("C", (), {"get": lambda self, s, k, d=None: d,
                         "data_dir": tmp_path})()
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"")
    counts = runner._crosscheck_figures(cfg, conn, {"id": "v", "duration_s": 600}, wav)

    assert loaded, "engine was never loaded — the failure path was not exercised"
    assert counts == {}
    assert conn.execute("SELECT crosscheck FROM number_ledger").fetchone()[0] is None


def test_a_disabled_crosscheck_is_a_no_op(conn, tmp_path):
    """The config switch must skip the work without touching any row."""
    from ytdigest import runner

    conn.execute("INSERT INTO videos (id, channel_id, title, discovered_at, status) "
                 "VALUES ('v','c','t','2026-01-01','normalized')")
    conn.execute("INSERT INTO number_ledger (video_id, raw_text, normalized, unit, start_s) "
                 "VALUES ('v','29億','2900000000','hkd',100.0)")
    conn.commit()

    cfg = type("C", (), {"get": lambda self, s, k, d=None: False,
                         "data_dir": tmp_path})()
    assert runner._crosscheck_figures(cfg, conn, {"id": "v", "duration_s": 600},
                                      tmp_path / "v.wav") == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_crosscheck.py -k dead_asr -v`
Expected: FAIL — `_crosscheck_figures` does not exist.

- [ ] **Step 3: Implement**

In `src/ytdigest/runner.py`, add above `stage_identify`:

```python
def _bias_terms(conn, video_id: str) -> str:
    """Qwen's context prompt, scoped to this video.

    Biasing is powerful and therefore dangerous: handed a global term list,
    Whisper turned 資金流 into 紫金流 purely because 紫金礦業 was in it. So this
    supplies the instruments this video actually mentions, not everything known.
    """
    from .views import load_instruments

    names = [r["instrument_raw"] for r in conn.execute(
        "SELECT DISTINCT instrument_raw FROM views WHERE video_id = ? "
        "AND instrument_raw IS NOT NULL LIMIT 30", (video_id,))]
    if not names:
        names = list(load_instruments().keys())[:20]
    return "以下是香港股評節目，涉及：" + "、".join(names) + "。"


def _crosscheck_figures(cfg, conn, video: dict, wav) -> dict[str, int]:
    """Give every figure a second reading. Never fatal.

    Runs after voice ID inside the same stage, sharing its wav -- audio is the
    expensive part and it is already on disk. Sequentially, never alongside:
    ASR peaks at 5.2 GB on a machine that is already swapping.
    """
    from .crosscheck import compare, spans_for, values_in
    from .normalize import _fmt

    if not cfg.get("asr", "crosscheck", True) or wav is None:
        return {}
    rows = list(conn.execute(
        "SELECT id, normalized, start_s FROM number_ledger "
        "WHERE video_id = ? AND start_s IS NOT NULL", (video["id"],)))
    if not rows:
        return {}

    try:
        engine = _load_asr(cfg)
        spans = spans_for([r["start_s"] for r in rows],
                          float(video.get("duration_s") or 0) or 1e9)
        from .interfaces import SpeechRegion
        segments = engine.transcribe(
            Path(wav), [SpeechRegion(lo, hi) for lo, hi in spans],
            lang_hint=None, context=_bias_terms(conn, video["id"]),
        )
    except Exception as exc:                    # noqa: BLE001 - never fatal
        log.warning("crosscheck.failed", video_id=video["id"], error=str(exc)[:200])
        return {}

    heard = [(s.start, s.end, values_in(s.text)) for s in segments]
    counts: dict[str, int] = {}
    with D.transaction(conn):
        for row in rows:
            near = [v for lo, hi, vals in heard
                    if lo <= row["start_s"] <= hi for v in vals]
            try:
                caption = float(row["normalized"]) if row["normalized"] else None
            except (TypeError, ValueError):
                caption = None
            state, rival = compare(caption, near)
            counts[state] = counts.get(state, 0) + 1
            # _fmt, not repr: the ledger's own `normalized` is written by _fmt,
            # which renders whole numbers as "2900000000". repr() would write
            # "29900000000.0", and Task 7 compares the two as strings.
            conn.execute(
                "UPDATE number_ledger SET crosscheck=?, asr_normalized=?, asr_model=? "
                "WHERE id=?",
                (state, _fmt(rival), engine.id, row["id"]),
            )
    log.info("crosscheck.done", video_id=video["id"], **counts)
    return counts
```

Then in `stage_identify`, replace its body's final two lines with:

```python
    from .voice import audio_for

    try:
        with audio_for(video["id"], cfg.data_dir / "audio") as wav:
            _identify_speakers(cfg, conn, video, segments, wav=wav)
            _crosscheck_figures(cfg, conn, video, wav)
    except Exception as exc:                    # noqa: BLE001 - never fatal
        log.warning("audio_stage.failed", video_id=video["id"], error=str(exc)[:200])
    return None
```

And change `_identify_speakers` to accept an already-downloaded `wav`, replacing
its own `with audio_for(...)` block with a direct call to `identify(conn, wav, ...)`.
Downloading twice would double the cost of the stage for no benefit.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_crosscheck.py -v`
Expected: PASS.

- [ ] **Step 5: Add the config switch**

In `config/config.toml` under `[asr]`:

```toml
crosscheck = true       # second reading of every ledger figure; false disables
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m pyflakes src/ytdigest tests`
Expected: all green.

- [ ] **Step 7: Run it on the video the bake-off used — the acceptance test**

The worktree's `data/state.db` is a snapshot of production carrying this video's
350 real ledger rows. Call the cross-check directly rather than `ytdigest retry`:
`retry` re-runs every stage including summarize, which costs a paid DeepSeek call
and re-publishes, neither of which this proves anything about.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from ytdigest.config import load_config
from ytdigest import db as D, runner
from ytdigest.voice import audio_for

cfg = load_config(); conn = D.connect(cfg.db_path)
video = dict(conn.execute("SELECT * FROM videos WHERE id='MgN00MCDDRM'").fetchone())
with audio_for(video["id"], cfg.data_dir / "audio") as wav:
    print("counts:", runner._crosscheck_figures(cfg, conn, video, wav))
for r in conn.execute(
    "SELECT raw_text, normalized, asr_normalized FROM number_ledger "
    "WHERE video_id='MgN00MCDDRM' AND crosscheck='disputed'"):
    print(dict(r))
PY
```

Expected: among the disputed rows, one whose `normalized` is `2900000000` and
`asr_normalized` is `29900000000`. **That is the acceptance test for this whole
plan.** If it does not appear, stop and report rather than continuing.

This downloads ~150 MB of audio and spends roughly 15-20 minutes of ASR on 29%
of a 2h31m video. Expect it to be slow; that is the measured cost, not a fault.

- [ ] **Step 8: Commit**

```bash
git add src/ytdigest/runner.py config/config.toml tests/test_crosscheck.py
git commit -m "Cross-check every figure inside the stage that already has the audio

Audio is the expensive part and stage_identify already downloads it for voice
ID, so the second reading costs CPU only. Sequential, never alongside: ASR
peaks at 5.2 GB on a machine already holding 8 of 16.

Bias terms are scoped to the video rather than global -- handed everything
known, Whisper turned 資金流 into 紫金流."
```

---

### Task 7: Refuse a disputed figure, and show both readings

**Files:**
- Modify: `src/ytdigest/validator.py:73-130`
- Modify: `src/ytdigest/runner.py` (stage_summarize's ledger load)
- Test: `tests/test_validator.py`

**Interfaces:**
- Consumes: `crosscheck.DISPUTED`, the `crosscheck`/`asr_normalized` columns.
- Produces: `validator.check_text(..., disputed: dict[str, str] | None = None)`, mapping a ledger `normalized` value to the rival reading. A summary figure resting on a disputed entry returns `verdict="flagged"` with reason `字幕係 X，我哋自己聽係 Y`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_validator.py`:

```python
def test_a_disputed_figure_cannot_verify_clean():
    """The captions put SMIC's inflow at 29億 and our own ASR at 299億. Until a
    human resolves that, the figure must not be published as verified -- which
    is exactly what happens today, because the ledger it is checked against was
    built from the same wrong transcript."""
    from ytdigest.normalize import LedgerEntry
    from ytdigest.validator import check_text

    ledger = [LedgerEntry(raw_text="29億", normalized="2900000000", unit="hkd",
                          segment_id=1, start_s=100.0, confidence=None, context="")]
    checks = check_text("北水淨流入29億", ledger,
                        disputed={"2900000000": "29900000000"})

    assert checks and checks[0].verdict == "flagged"
    assert "29900000000" in checks[0].reason


def test_an_agreed_figure_still_passes():
    from ytdigest.normalize import LedgerEntry
    from ytdigest.validator import check_text

    ledger = [LedgerEntry(raw_text="56億", normalized="5600000000", unit="hkd",
                          segment_id=1, start_s=100.0, confidence=None, context="")]
    checks = check_text("淨流入56億", ledger, disputed={})
    assert checks and checks[0].verdict == "ok"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_validator.py -k disputed -v`
Expected: FAIL — `check_text() got an unexpected keyword argument 'disputed'`.

- [ ] **Step 3: Implement**

Add the parameter to `check_text` with a default of `None`, and immediately
before each `verdict="ok"` return path insert:

```python
        # A figure our own ASR heard differently is not verified, whatever the
        # ledger says -- the ledger and the summary were both built from the
        # same transcript, so agreeing with each other proves nothing about
        # what was actually said.
        rival = (disputed or {}).get(entry.normalized)
        if rival is not None:
            checks.append(Check(
                figure=figure, unit=unit, value=value, verdict="flagged",
                reason=f"字幕係 {entry.normalized}，我哋自己聽係 {rival}",
                ledger_start_s=entry.start_s, confidence=entry.confidence,
            ))
            continue
```

In `runner.stage_summarize`, build the map alongside the ledger it already loads:

```python
    disputed = {
        r["normalized"]: r["asr_normalized"]
        for r in conn.execute(
            "SELECT normalized, asr_normalized FROM number_ledger "
            "WHERE video_id = ? AND crosscheck = 'disputed'", (video["id"],))
        if r["normalized"]
    }
```

and pass `disputed=disputed` into the validator call.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_validator.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm the digest already surfaces it**

Run: `.venv/bin/python -m pytest tests/test_telegram.py -v`
Expected: PASS unchanged. A disputed figure returns `flagged`, and
`render_telegram` already marks anything that is not `ok` with `⚠︎` and prints
the legend — no rendering change is needed.

- [ ] **Step 6: Full suite and pyflakes**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m pyflakes src/ytdigest tests`
Expected: all green.

- [ ] **Step 7: Update the durable notes**

In `AGENTS.md`, under the section added on 2026-08-11, append:

```markdown
Since 2026-08-11 the audio stage gives every ledger figure a second reading
from Qwen3-ASR and records `agreed` / `disputed` / `absent` / `unchecked` on
`number_ledger`. A disputed figure returns `flagged`, so the validator refuses
it and the digest marks it. **`crosscheck IS NULL` means never checked and must
never be read as agreement** — every row written before that date is null.
Expect more flags than before: it is the first time anything could tell.
```

- [ ] **Step 8: Commit**

```bash
git add src/ytdigest/validator.py src/ytdigest/runner.py tests/test_validator.py AGENTS.md
git commit -m "Refuse a figure our own ASR heard differently

The ledger and the summary are both built from the same transcript, so their
agreeing with each other proves nothing about what was said. A figure with a
second reading that disagrees now returns flagged, carrying both numbers, and
the digest already renders that path.

Digests will show more warnings than before. That is the first time anything
has been able to tell."
```

---

## Self-review

**Spec coverage.** Architecture (one engine, spans mode) → Task 3. 29% windows → Task 5 `spans_for`, verified in Task 6. Ledger columns and the four states → Tasks 4 and 5. `absent` vs `disputed` → Task 5, two dedicated tests. Compare on `normalized` not `raw_text` → Task 5 and Task 7's `disputed` map, both keyed on `normalized`. Disputed → `flagged` → `⚠︎` with no new rendering path → Task 7. Never fail a video → Task 3 step 1 test, Task 6 step 1 test, and try/except in both. Per-video bias list → Task 6 `_bias_terms`. Model id correction → Task 3 step 6. Sequential ASR/voice → Task 6 step 3. Numeral parser prerequisite → Tasks 1 and 2.

**Gap found and closed:** the spec assumed full-file mode was a small variation on spans mode. The bake-off showed Qwen returns one untimed segment for its whole input, so full mode needs VAD first. Rather than leave it implicit, Task 3 raises `NotImplementedError` with a test pinning that behaviour, and the "Out of scope" section records why Ron's channel is a separate plan.

**Placeholders:** none. Every code step carries the actual code.

**Type consistency:** `compare` returns `tuple[str, float | None]` in Task 5 and is unpacked as `state, rival` in Task 6. `spans_for` returns `list[tuple[float, float]]`, converted to `SpeechRegion` in Task 6 before reaching `transcribe`, which takes `list[SpeechRegion]` in Task 3.

One mismatch was found here and fixed inline: Task 6 originally stored the rival reading with `repr()`, giving `"29900000000.0"`, while the ledger's own `normalized` column is written by `normalize._fmt`, which renders whole numbers as `"2900000000"`. Task 7 compares those two as strings, so every dispute would have been recorded and then failed to match. Task 6 now uses `_fmt` on both sides.
