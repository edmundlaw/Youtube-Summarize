"""The transcript contract (ARCHITECTURE §5).

The most important schema in the system. Everything downstream depends on it,
and it must survive ASR model swaps. An unknown schema_version is refused
loudly rather than guessed at.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .interfaces import TRANSCRIPT_SCHEMA_VERSION, Segment


class SchemaVersionError(RuntimeError):
    """Raised on an unknown transcript schema_version.

    Carries error_class so the runner classifies it permanent: retrying a
    version mismatch three times cannot help, and the generic handler would
    otherwise treat it as retryable.
    """

    error_class = "permanent"


def write_transcript(
    path: Path,
    video_id: str,
    segments: list[Segment],
    *,
    source: str,
    model_id: str | None = None,
    params_hash: str | None = None,
    audio: dict | None = None,
) -> Path:
    confidences = [s.confidence for s in segments if s.confidence is not None]
    payload = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "video_id": video_id,
        "source": source,
        "model": {"id": model_id, "params_hash": params_hash},
        "audio": audio or {},
        "dominant_lang": None,  # set by normalize, which does the language work
        "segment_count": len(segments),
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "segments": [s.to_dict() for s in segments],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_transcript(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != TRANSCRIPT_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{path.name}: schema_version {version!r}, this build understands "
            f"{TRANSCRIPT_SCHEMA_VERSION}. Refusing to guess."
        )
    data["segments"] = [Segment(**s) for s in data["segments"]]
    return data


def transcript_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
