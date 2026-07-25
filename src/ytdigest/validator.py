"""Post-hoc validation of generated figures against the number ledger.

Three real failures from live runs drive this design. Each has a test.

1. FABRICATION. Captions truncated mid-sentence ("所以應該係八、二…") and the
   model completed the thought with an invented HSI target of 28243. Nothing in
   the prompt permitted this; the prompt already said "copy, never recompute".
   Prompt discipline is not a control. Only comparison against the ledger is.

2. UNIT STRIPPING. An earlier hand-rolled check compared digits only, so the
   generated "2000億" was matched against a stray "2000" elsewhere in the
   transcript and passed — while the phrase "2000億" appeared nowhere. Matching
   must be on value AND unit together.

3. SPLIT NUMBER. Rolling captions interleaved two positions and separated a
   figure from its unit: "已經有4200裏邊咧…其億美金". The figure is genuinely
   present but the ledger row for it is a bare COUNT. Rejecting the model's
   correct "4200億美金" would punish it for the transcript's defect, so a
   value match with a compatible unit downgrades to a flag rather than a fail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import LedgerEntry
from .numbers import COUNT, HKD, USD, CNY, find_numbers, is_financially_meaningful

PASSED = "passed"
PASSED_WITH_FLAGS = "passed_with_flags"
FAILED = "failed"

#: Units that may legitimately stand in for one another, downgrading a
#: mismatch from "fabricated" to "verify this one".
#:
#: Two real reasons, both observed:
#:  * caption corruption strips the currency word, leaving a bare magnitude
#:    ("已經有4200裏邊咧…其億美金");
#:  * spoken Cantonese uses 蚊 for dollars generically, so a US-listed stock
#:    quoted as "223蚊" is correctly rendered "223美元" by the summariser.
#:    Failing that outright cries wolf on a correct figure.
_MONEY_UNITS = {USD, CNY, HKD}
_COMPATIBLE: dict[str, set[str]] = {
    **{u: (_MONEY_UNITS - {u}) | {COUNT} for u in _MONEY_UNITS},
    COUNT: set(_MONEY_UNITS),
}

#: Relative tolerance. Zero by default: a summary figure must be the ledger
#: figure, not a rounding of it.
TOLERANCE = 0.0


@dataclass
class Check:
    figure: str
    unit: str
    value: float | None
    verdict: str          # ok | flagged | missing
    reason: str
    ledger_start_s: float | None = None
    confidence: float | None = None


def _matches(value: float | None, target: float | None) -> bool:
    if value is None or target is None:
        return False
    if TOLERANCE == 0:
        return value == target
    return abs(value - target) <= abs(target) * TOLERANCE


def check_text(
    text: str,
    ledger: list[LedgerEntry],
    low_confidence: float = 0.55,
) -> list[Check]:
    """Validate every numeric mention in `text` against the ledger."""
    checks: list[Check] = []
    for hit in find_numbers(text):
        # Clock times and moving-average labels are numerals, not claims. The
        # ledger does not hold them, so validating them would fail every
        # summary that mentions when the show starts.
        if not is_financially_meaningful(hit.unit):
            continue
        exact = [
            e for e in ledger
            if e.unit == hit.unit and _matches(hit.value, _num(e.normalized))
        ]
        if exact:
            best = exact[0]
            low = best.confidence is not None and best.confidence < low_confidence
            checks.append(
                Check(
                    figure=hit.raw, unit=hit.unit, value=hit.value,
                    verdict="flagged" if low else "ok",
                    reason="source segment below confidence threshold" if low
                           else "matched ledger on value and unit",
                    ledger_start_s=best.start_s, confidence=best.confidence,
                )
            )
            continue

        compatible = [
            e for e in ledger
            if e.unit in _COMPATIBLE.get(hit.unit, set())
            and _matches(hit.value, _num(e.normalized))
        ]
        if compatible:
            best = compatible[0]
            checks.append(
                Check(
                    figure=hit.raw, unit=hit.unit, value=hit.value,
                    verdict="flagged",
                    reason=f"value matched but unit differs (ledger: {best.unit}) — "
                           "likely a transcript defect, verify manually",
                    ledger_start_s=best.start_s, confidence=best.confidence,
                )
            )
            continue

        checks.append(
            Check(
                figure=hit.raw, unit=hit.unit, value=hit.value,
                verdict="missing",
                reason="no ledger entry with this value and unit — treat as unverified",
            )
        )
    return checks


def verdict(checks: list[Check]) -> str:
    if any(c.verdict == "missing" for c in checks):
        return FAILED
    if any(c.verdict == "flagged" for c in checks):
        return PASSED_WITH_FLAGS
    return PASSED


def offending(checks: list[Check]) -> list[str]:
    return [c.figure for c in checks if c.verdict == "missing"]


def render_figure(check: Check) -> str:
    """Figures that did not cleanly verify are always visually marked."""
    return check.figure if check.verdict == "ok" else f"⚠︎{check.figure}"


def annotate(text: str, checks: list[Check]) -> str:
    """Mark unverified figures in place, longest-first so shorter figures that
    are substrings of longer ones do not corrupt the replacement."""
    marked = sorted(
        {c.figure for c in checks if c.verdict != "ok"}, key=len, reverse=True
    )
    for figure in marked:
        text = re.sub(
            r"(?<!⚠︎)" + re.escape(figure), "⚠︎" + figure, text, count=0
        )
    return text


def retry_instruction(checks: list[Check], ledger: list[LedgerEntry]) -> str:
    """Name the offending figures explicitly for the regeneration pass."""
    bad = [c for c in checks if c.verdict == "missing"]
    if not bad:
        return ""
    lines = [
        "你上一次輸出裏面有以下數字，喺字幕原文搵唔到，屬於捏造或推算：",
        *[f"  - {c.figure}（{c.reason}）" for c in bad],
        "",
        "重寫一次。以下係字幕入面真正出現過嘅數字，你只可以用呢啲：",
    ]
    seen: set[str] = set()
    for entry in ledger[:80]:
        key = f"{entry.raw_text}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  - {entry.raw_text}  [{int(entry.start_s)//60:02d}:"
                     f"{int(entry.start_s)%60:02d}]")
    lines.append("")
    lines.append("如果某個講法喺字幕度斷咗、殘缺、講到一半冇咗，就唔好寫個數字，"
                 "改為寫「字幕於此中斷」。寧願少寫，都唔可以砌一個出嚟。")
    return "\n".join(lines)


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None
