"""WebVTT parsing, including YouTube's rolling auto-captions.

YouTube's auto-caption VTT is not a clean list of utterances. Cues overlap and
repeat: each cue re-displays the tail of the previous one, then appends new
words. Naive text dedup produces output like

    ...因為我哋冇嗰啲晶片。呀，冇個算力，而家個算力呢？而家只係用如果冇個算力...

where whole clauses appear two or three times. That corrupts the number ledger
(one figure counted many times) and wastes summariser tokens.

The fix is to stop treating this as a string problem. Auto-caption cues carry
*per-character inline timestamps*:

    個<00:00:47.399><c>交</c><00:00:47.600><c>易</c><00:00:47.760><c>日</c>

Every character therefore has a unique authoritative time. Collect (time, char)
pairs across the whole file, keep the first occurrence of each timestamp, and
sort. Repetition disappears by construction rather than by heuristic, and the
result is exactly reconstructible from the source.

Files without inline timings (human-authored subtitles) take the plain path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .interfaces import Segment

_CUE_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)
_INLINE_TS = re.compile(r"<(\d{2}):(\d{2}):(\d{2})\.(\d{3})>")
_TAG = re.compile(r"</?c[^>]*>|<[^>]+>")
_HEADER = ("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


@dataclass
class Cue:
    start: float
    end: float
    payload: str  # raw, tags intact


def parse_cues(vtt_text: str) -> list[Cue]:
    cues: list[Cue] = []
    current: Cue | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal current, lines
        if current is not None:
            current.payload = "\n".join(lines).strip()
            if current.payload:
                cues.append(current)
        current, lines = None, []

    for raw_line in vtt_text.splitlines():
        match = _CUE_TIME.search(raw_line)
        if match:
            flush()
            current = Cue(
                start=_seconds(*match.group(1, 2, 3, 4)),
                end=_seconds(*match.group(5, 6, 7, 8)),
                payload="",
            )
            continue
        if current is None:
            continue
        if raw_line.startswith(_HEADER):
            continue
        lines.append(raw_line)
    flush()
    return cues


def _timed_characters(cues: list[Cue]) -> list[tuple[float, str]]:
    """Explode cues into (timestamp, text) pairs using inline word timings.

    The structure of a rolling auto-caption cue is per *line*, not per cue:

        咦，大師咁早嘅？中早，9點半開始啦。每      <- carry-over from earlier cues
        個<00:00:47.399><c>交</c><00:00:47.600><c>易</c>   <- the new words

    Only the line bearing inline timestamps is new; every line above it is a
    redisplay of text already emitted. Attributing that carry-over to the cue's
    own start time (as an earlier version of this function did) gives each
    repetition a distinct key, so nothing dedupes and the transcript comes out
    roughly 3x too long — worse than naive string matching.

    So: drop the carry-over lines outright, and key the remaining text by its
    inline timestamp, which is stable across redisplays. The short run of text
    before the first inline timestamp on the active line is genuine new content;
    it is keyed just under that timestamp so it sorts into place and still
    collapses on repeat.
    """
    seen: dict[float, str] = {}
    for cue in cues:
        for line in cue.payload.splitlines():
            if not _INLINE_TS.search(line):
                continue  # carry-over, or a settled duplicate of new text
            parts = _INLINE_TS.split(line)
            # split() yields: [text, h, m, s, ms, text, h, m, s, ms, ...]
            first_stamp = _seconds(*parts[1:5])
            head = _TAG.sub("", parts[0]).strip()
            if head:
                seen.setdefault(round(first_stamp - 0.001, 3), head)
            index = 1
            while index + 4 <= len(parts):
                stamp = _seconds(*parts[index : index + 4])
                chunk = _TAG.sub("", parts[index + 4]) if index + 4 < len(parts) else ""
                if chunk:
                    seen.setdefault(stamp, chunk)
                index += 5
    return sorted(seen.items())


def _dedupe_plain(cues: list[Cue]) -> list[Cue]:
    """Fallback for captions with no inline timings.

    Drops a cue whose text is wholly contained in the previous one, and trims a
    repeated prefix when a cue extends its predecessor.
    """
    out: list[Cue] = []
    for cue in cues:
        text = _TAG.sub("", cue.payload).strip()
        if not text:
            continue
        if out:
            previous = out[-1].payload
            if text == previous or text in previous:
                continue
            if text.startswith(previous):
                text = text[len(previous) :].strip()
                if not text:
                    continue
        out.append(Cue(cue.start, cue.end, text))
    return out


def _group(
    pairs: list[tuple[float, str]], max_gap: float, max_chars: int
) -> list[tuple[float, float, str]]:
    """Group timed characters into segment-sized spans."""
    spans: list[tuple[float, float, str]] = []
    if not pairs:
        return spans
    start = pairs[0][0]
    previous = start
    buffer: list[str] = []
    for stamp, text in pairs:
        too_long = len(" ".join(buffer)) >= max_chars
        if buffer and (stamp - previous > max_gap or too_long):
            spans.append((start, previous, "".join(buffer).strip()))
            buffer, start = [], stamp
        buffer.append(text)
        previous = stamp
    if buffer:
        spans.append((start, previous, "".join(buffer).strip()))
    return [s for s in spans if s[2]]


def vtt_to_segments(
    vtt_text: str,
    *,
    max_gap: float = 2.0,
    max_chars: int = 120,
) -> list[Segment]:
    """Parse a VTT file into deduplicated transcript segments.

    Subtitle-sourced segments carry no confidence: the source does not report
    one, and inventing a number here would let downstream code believe a figure
    was verified when it was not.
    """
    cues = parse_cues(vtt_text)
    if not cues:
        return []

    pairs = _timed_characters(cues)
    if pairs:
        spans = _group(pairs, max_gap=max_gap, max_chars=max_chars)
        # Some utterances never appear on an inline-timestamped line at all —
        # a short standalone "好" between topics, for instance. Dropping every
        # plain cue would lose them silently, so fold back any whose text the
        # timed reconstruction does not already contain.
        # Plain cues roll against each other too, so collapse them among
        # themselves first; only then ask whether what remains is genuinely
        # absent from the timed reconstruction.
        covered = "".join(text for _, _, text in spans)
        plain = _dedupe_plain([c for c in cues if not _INLINE_TS.search(c.payload)])
        recovered: list[tuple[float, float, str]] = []
        for cue in plain:
            text = cue.payload.replace("\n", "").strip()
            if not text or text in covered:
                continue
            covered += text
            recovered.append((cue.start, cue.end, text))
        if recovered:
            spans = sorted(spans + recovered, key=lambda s: s[0])
    else:
        spans = [(c.start, c.end, c.payload) for c in _dedupe_plain(cues)]

    return [
        Segment(id=i, start=start, end=end, text=text, confidence=None)
        for i, (start, end, text) in enumerate(spans)
    ]


def is_usable_subtitle_track(name: str) -> bool:
    """`live_chat` is offered alongside real subtitle tracks but is a JSON chat
    replay, not a transcript. Treating it as one produces a 'transcript' of
    viewer comments — silently, and it looks plausible until you read it."""
    return name not in {"live_chat"}
