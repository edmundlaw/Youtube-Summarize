"""Normalisation and number-ledger extraction.

Runs before any LLM sees the transcript. What this module writes into
`number_ledger` is the authority against which every generated figure is
checked; the LLM is never trusted to recall or recompute a number.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .interfaces import Segment
from .numbers import find_numbers, is_financially_meaningful

# Codepoints that exist only in Simplified. Presence of these means the text is
# Simplified-dominant and worth converting; absence means leave it alone.
# Running s2hk over already-Traditional text mangles it.
_SIMPLIFIED_ONLY = set("们义产众优会传体价众关兴写农冲决击刘则务动势区华协单卖")

_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[一-鿿]")


def detect_script(text: str) -> str:
    """'simplified' | 'traditional' | 'none'."""
    if not _CJK.search(text):
        return "none"
    return "simplified" if any(c in _SIMPLIFIED_ONLY for c in text) else "traditional"


def to_hk_traditional(text: str) -> str:
    """Convert Simplified to HK Traditional, but only when it is Simplified.

    Colloquial Cantonese (嘅 咗 喺 唔 哋 嘢) passes through untouched — OpenCC's
    s2hk does not standardise it, and we must not either. What the speaker said
    is information; 'correcting' it to written Chinese loses meaning.
    """
    if detect_script(text) != "simplified":
        return text
    try:
        from opencc import OpenCC

        return OpenCC("s2hk").convert(text)
    except Exception:
        return text


def guess_lang(text: str) -> str | None:
    """Per-segment language tag. Never per file — HK finance code-switches
    constantly and a single language flag destroys that."""
    if not text.strip():
        return None
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk == 0 and latin > 0:
        return "en"
    if cjk == 0:
        return None
    # Cantonese-only particles. Their presence is decisive; their absence is not.
    if any(c in text for c in "嘅咗喺唔哋嘢乜嗰咁啦㗎喇咩"):
        return "yue"
    return "zh"


@dataclass
class LedgerEntry:
    raw_text: str
    normalized: str | None
    unit: str
    segment_id: int
    start_s: float
    confidence: float | None
    context: str

    def as_row(self, video_id: str) -> tuple:
        return (
            video_id, self.raw_text, self.normalized, self.unit,
            self.segment_id, self.start_s, self.confidence, self.context,
        )


# NOTE: structural promo detection was tried and removed. RagaFinance airs
# trailers for its other shows mid-episode, and those trailers name a different
# show's host. Detecting them by verbatim repetition does not work: the ASR
# transcribes the same advert differently each airing — on a real file the two
# airings shared only 6 identical characters. Attribution is constrained at the
# prompt level instead, from the host list in the video title. See
# summarize.hosts_from_title().


def normalize_segments(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        text = to_hk_traditional(seg.text)
        out.append(
            Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=text,
                lang=seg.lang or guess_lang(text),
                confidence=seg.confidence,
                flags=list(seg.flags),
            )
        )
    return out


def dominant_lang(segments: list[Segment]) -> str:
    counts: dict[str, int] = {}
    for seg in segments:
        if seg.lang:
            counts[seg.lang] = counts.get(seg.lang, 0) + len(seg.text)
    if not counts:
        return "unknown"
    top, total = max(counts.items(), key=lambda kv: kv[1]), sum(counts.values())
    # No clear majority in a code-switched file -> treat as mixed, which the
    # output-language rule maps to Traditional Chinese anyway.
    return top[0] if top[1] / total >= 0.5 else "mixed"


def build_ledger(segments: list[Segment], context_window: int = 1) -> list[LedgerEntry]:
    """Extract every financially meaningful figure with provenance.

    Each entry carries the timestamp, the confidence of its source segment and
    a window of surrounding text, so a downstream reviewer can jump straight to
    the moment a figure was spoken.
    """
    entries: list[LedgerEntry] = []
    for index, seg in enumerate(segments):
        lo = max(0, index - context_window)
        hi = min(len(segments), index + context_window + 1)
        context = "".join(s.text for s in segments[lo:hi])
        for hit in find_numbers(seg.text):
            if not is_financially_meaningful(hit.unit):
                continue
            entries.append(
                LedgerEntry(
                    raw_text=hit.raw,
                    normalized=_fmt(hit.value),
                    unit=hit.unit,
                    segment_id=seg.id,
                    start_s=seg.start,
                    confidence=seg.confidence,
                    context=context[:500],
                )
            )
    return entries


def _fmt(value: float | None) -> str | None:
    if value is None:
        return None
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def write_normalized(
    path: Path, video_id: str, segments: list[Segment], ledger: list[LedgerEntry]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "dominant_lang": dominant_lang(segments),
        "segments": [s.to_dict() for s in segments],
        "ledger": [asdict(e) for e in ledger],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
