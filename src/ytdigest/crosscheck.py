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


def _digits(v: float) -> str:
    """Bare digit string for edit-distance comparison, e.g. 2_900_000_000 ->
    "2900000000". Fixed-point, never scientific notation, so magnitude is not
    lost for the billion-scale figures this module exists to check."""
    return f"{abs(round(v)):.0f}"


def _levenshtein(a: str, b: str) -> int:
    """Single-digit insertions/deletions/substitutions separating two strings."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _nearest(caption_value: float, asr_values: list[float]) -> float:
    """The rival reading most likely to be the *same* figure misheard.

    Raw numeric distance is the wrong ruler here: on the case this module was
    built for (MgN00MCDDRM @7483s), captions hold 2,900,000,000 (29億) and ASR
    offers both 29,900,000,000 (299億, the true misreading -- one inserted
    digit away) and 5,600,000,000 (56億, an unrelated figure for a different
    stock mentioned in the same breath). By absolute value 56億 is ten times
    closer, yet it is two digits away by substitution while 299億 is one
    digit away by insertion. ASR errors are digit-level, not additive noise,
    so digit edit-distance -- not arithmetic distance -- is what separates
    "the same number misheard" from "a different number that happens to sit
    nearby."
    """
    caption_digits = _digits(caption_value)
    return min(asr_values, key=lambda v: _levenshtein(caption_digits, _digits(v)))


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
    return DISPUTED, _nearest(caption_value, asr_values)
