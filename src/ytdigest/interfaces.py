"""The three seams where implementations get swapped.

Nothing outside an adapter module may import a model library. If you find
`import mlx` in pipeline code, that is the bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

TRANSCRIPT_SCHEMA_VERSION = 1


@dataclass
class SpeechRegion:
    """A contiguous span of speech found by VAD, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Segment:
    """One transcript segment. `confidence` is normalised to 0-1 by the adapter,
    whatever the underlying model reports. `lang` is per segment, never per file."""

    id: int
    start: float
    end: float
    text: str
    lang: str | None = None
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "lang": self.lang,
            "confidence": self.confidence,
            "flags": self.flags,
        }


@dataclass
class VideoRef:
    id: str
    channel_id: str
    title: str
    published_at: str
    duration_s: int | None = None


@dataclass
class FetchResult:
    """What `fetch` produced. Exactly one of audio_path / subtitle_path is set."""

    kind: str  # "audio" | "manual_subs" | "auto_subs"
    path: Path
    duration_s: float | None = None


class ASREngine(Protocol):
    id: str

    def params_hash(self) -> str: ...

    def transcribe(
        self,
        audio_path: Path,
        chunks: list[SpeechRegion],
        lang_hint: str | None = None,
    ) -> list[Segment]: ...


class Summarizer(Protocol):
    id: str

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class SourceAdapter(Protocol):
    def discover(self, channel) -> list[VideoRef]: ...

    def fetch(self, video: VideoRef, dest: Path) -> FetchResult: ...
