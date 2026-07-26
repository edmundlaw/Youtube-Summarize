"""The sequential runner and the individual pipeline stages.

Two structural rules, both load-bearing on a 16 GB machine shared with a dozen
other services:

* Single instance, enforced by flock. launchd firing while a long transcription
  runs must be a no-op, not a crash or a second model in memory.
* Every heavy stage runs as a separate subprocess. MLX and Python do not
  reliably return unified memory to the OS within a process, so a fresh process
  per stage is the only guarantee the ASR model is gone before the summariser
  starts.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import db as D
from .db import reap_orphan_runs as db_reap
from .config import Config
from .logging import get_logger

log = get_logger()


@contextmanager
def single_instance(lock_path: Path):
    """Yield True if we hold the lock, False if another run already does."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        handle.write(str(__import__("os").getpid()))
        handle.flush()
        yield True
    finally:
        handle.close()


# --- stages ----------------------------------------------------------------


def stage_fetch(cfg: Config, conn, video: dict) -> tuple[str, Path]:
    from .sources import youtube

    row = conn.execute(
        "SELECT lang_hint, min_duration_s, max_duration_s FROM channels WHERE id=?",
        (video["channel_id"],),
    ).fetchone()
    lang_hint = row["lang_hint"] if row else None

    result = youtube.fetch(video["id"], cfg.data_dir / "audio", lang_hint)

    duration = result.duration_s or 0
    if row and duration:
        if duration < (row["min_duration_s"] or 0):
            raise D.StageError(f"too short ({duration:.0f}s)", D.PERMANENT)
        if duration > (row["max_duration_s"] or 10**9):
            raise D.StageError(f"too long ({duration:.0f}s)", D.PERMANENT)
        conn.execute(
            "UPDATE videos SET duration_s=? WHERE id=?", (int(duration), video["id"])
        )
    return result.kind, result.path


def stage_transcribe(cfg: Config, conn, video: dict) -> Path:
    """Build the schema-versioned transcript from whatever fetch produced."""
    from .transcript import write_transcript
    from .vtt import vtt_to_segments

    artifact = D.get_artifact(conn, video["id"], "audio")
    if artifact is None:
        raise D.StageError("no fetched artifact", D.RETRYABLE)
    source_path = Path(artifact["path"])

    if source_path.suffix == ".vtt":
        segments = vtt_to_segments(source_path.read_text(encoding="utf-8"))
        # Read the kind the fetch stage recorded. Deriving it from the filename
        # published auto-generated captions as `manual_subs`.
        kind = (artifact["meta"] or "auto_subs") if "meta" in artifact.keys() else "auto_subs"
        model_id = None
    else:
        engine = _load_asr(cfg)
        segments = engine.transcribe(source_path, [], None)
        kind, model_id = "asr", engine.id

    if not segments:
        raise D.StageError("transcript is empty", D.RETRYABLE)

    out = cfg.data_dir / "transcripts" / f"{video['id']}.json"
    write_transcript(out, video["id"], segments, source=kind, model_id=model_id)

    return out


def _load_asr(cfg: Config):
    engine = cfg.get("asr", "engine", "qwen3")
    if engine == "qwen3":
        try:
            from .asr.qwen3 import Qwen3ASRMLX
        except ImportError as exc:
            # M2 is deliberately unbuilt: auto-captions cover every video seen
            # so far. A video with no subtitles must fail with a message that
            # says so, not a bare ModuleNotFoundError.
            raise D.StageError(
                "this video has no usable subtitles and the ASR engine is not "
                "built (M2). Install extras and implement ytdigest.asr.qwen3, "
                f"or skip this video. ({exc})",
                D.PERMANENT,
            ) from exc
        return Qwen3ASRMLX(cfg)
    raise D.StageError(f"unknown ASR engine {engine!r}", D.PERMANENT)


def stage_normalize(cfg: Config, conn, video: dict) -> Path:
    from .normalize import build_ledger, dominant_lang, normalize_segments, write_normalized
    from .transcript import read_transcript

    artifact = D.get_artifact(conn, video["id"], "transcript")
    if artifact is None:
        raise D.StageError("no transcript artifact", D.RETRYABLE)

    data = read_transcript(Path(artifact["path"]))
    segments = normalize_segments(data["segments"])
    ledger = build_ledger(segments)

    with D.transaction(conn):
        conn.execute("DELETE FROM number_ledger WHERE video_id=?", (video["id"],))
        conn.executemany(
            "INSERT INTO number_ledger (video_id, raw_text, normalized, unit, "
            "segment_id, start_s, confidence, context) VALUES (?,?,?,?,?,?,?,?)",
            [e.as_row(video["id"]) for e in ledger],
        )
        conn.execute(
            "INSERT INTO transcripts (video_id, schema_version, source, model_id, "
            "params_hash, dominant_lang, segment_count, mean_confidence, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(video_id) DO UPDATE SET "
            "dominant_lang=excluded.dominant_lang, segment_count=excluded.segment_count",
            (
                video["id"], data["schema_version"], data["source"],
                data.get("model", {}).get("id"), data.get("model", {}).get("params_hash"),
                dominant_lang(segments), len(segments),
                _mean_confidence(segments), D.now_iso(),
            ),
        )

    out = cfg.data_dir / "normalized" / f"{video['id']}.json"
    write_normalized(out, video["id"], segments, ledger)
    log.info("normalize.ledger", video_id=video["id"], entries=len(ledger))
    return out


def _mean_confidence(segments) -> float | None:
    values = [s.confidence for s in segments if s.confidence is not None]
    return sum(values) / len(values) if values else None


def stage_summarize(cfg: Config, conn, video: dict) -> Path:
    from .normalize import LedgerEntry
    from .summarize import PROMPT_VERSION, summarize
    from .transcript import read_transcript

    artifact = D.get_artifact(conn, video["id"], "normalized")
    if artifact is None:
        raise D.StageError("no normalized artifact", D.RETRYABLE)
    data = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))

    from .interfaces import Segment

    segments = [Segment(**s) for s in data["segments"]]
    ledger = [LedgerEntry(**e) for e in data["ledger"]]

    transcript_artifact = D.get_artifact(conn, video["id"], "transcript")
    payload, state, checks = summarize(
        cfg, segments, ledger, data.get("dominant_lang", "yue"), log
    )

    out = cfg.data_dir / "normalized" / f"{video['id']}.summary.json"
    out.write_text(
        json.dumps(
            {
                "payload": payload,
                "state": state,
                "checks": [c.__dict__ for c in checks],
                "prompt_version": PROMPT_VERSION,
                "summary_model": cfg.get("summarize", "model"),
                "transcript_sha": transcript_artifact["sha256"] if transcript_artifact else None,
                "lang": data.get("dominant_lang"),
                "source": read_transcript(Path(transcript_artifact["path"]))["source"]
                if transcript_artifact else None,
            },
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    log.info("summarize.done", video_id=video["id"], validator=state,
             figures=len(checks))
    return out


def stage_publish(cfg: Config, conn, video: dict) -> Path:
    from .publish import render_markdown, render_telegram, send_telegram, write_markdown
    from .validator import Check

    artifact = D.get_artifact(conn, video["id"], "summary")
    if artifact is None:
        raise D.StageError("no summary artifact", D.RETRYABLE)
    data = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    checks = [Check(**c) for c in data["checks"]]

    channel = conn.execute(
        "SELECT title FROM channels WHERE id=?", (video["channel_id"],)
    ).fetchone()
    view = dict(video)
    view["channel_title"] = channel["title"] if channel else ""

    markdown = render_markdown(
        video=view, payload=data["payload"], checks=checks, state=data["state"],
        meta={
            "lang": data.get("lang"),
            "source": data.get("source"),
            "prompt_version": data.get("prompt_version"),
            "summary_model": data.get("summary_model"),
            "transcript_sha": data.get("transcript_sha"),
        },
    )
    path = write_markdown(cfg.out_dir, view, markdown)

    if cfg.get("publish", "telegram", False):
        send_telegram(cfg, render_telegram(view, data["payload"], checks, data["state"]))

    with D.transaction(conn):
        conn.execute(
            "INSERT INTO summaries (video_id, prompt_version, model_id, "
            "transcript_sha, validator_state, path, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                video["id"], data.get("prompt_version", "?"),
                data.get("summary_model", "?"), data.get("transcript_sha", ""),
                data["state"], str(path), D.now_iso(),
            ),
        )
    return path


STAGE_FUNCS = {
    "fetch": stage_fetch,
    "transcribe": stage_transcribe,
    "normalize": stage_normalize,
    "summarize": stage_summarize,
    "publish": stage_publish,
}

ARTIFACT_KIND = {
    "fetch": "audio",
    "transcribe": "transcript",
    "normalize": "normalized",
    "summarize": "summary",
    "publish": None,
}


def run_stage_inprocess(cfg: Config, conn, video_id: str, stage: str) -> None:
    """Execute one stage and commit its outcome. Called in the child process."""
    video = dict(conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone())
    attempt = D.attempts_for(conn, video_id, stage) + 1
    run_id = D.start_stage(conn, video_id, stage, attempt)
    started = time.monotonic()
    try:
        result = STAGE_FUNCS[stage](cfg, conn, video)
        elapsed = int((time.monotonic() - started) * 1000)
        kind = ARTIFACT_KIND[stage]
        artifact = meta = None
        if kind is not None:
            if isinstance(result, tuple):
                meta, path = result          # fetch returns (source_kind, path)
            else:
                path = result
            artifact = (kind, Path(path))
        D.finish_stage_ok(conn, run_id, video_id, stage, elapsed, artifact, meta)
        log.info("stage.ok", video_id=video_id, stage=stage, duration_ms=elapsed)
    except D.StageError as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        status = D.finish_stage_failed(
            conn, run_id, video_id, stage, elapsed, exc.error_class, str(exc),
            int(cfg.get("runner", "max_attempts", 3)),
        )
        log.error("stage.failed", video_id=video_id, stage=stage,
                  duration_ms=elapsed, error_class=exc.error_class,
                  error=str(exc)[:300], status=status)
        raise
    except Exception as exc:  # unexpected: keep the trace, honour any hint
        elapsed = int((time.monotonic() - started) * 1000)
        status = D.finish_stage_failed(
            conn, run_id, video_id, stage, elapsed,
            getattr(exc, "error_class", D.RETRYABLE),
            f"{type(exc).__name__}: {exc}",
            int(cfg.get("runner", "max_attempts", 3)),
        )
        log.exception("stage.crashed", video_id=video_id, stage=stage, status=status)
        raise


def _timeout_for(cfg: Config, stage: str) -> int:
    return int(cfg.get("runner", f"timeout_{stage}_s", 900))


def run_stage_subprocess(cfg: Config, video_id: str, stage: str) -> bool:
    """Spawn `ytdigest _stage` so heavy memory is reclaimed on exit."""
    command = [sys.executable, "-m", "ytdigest.cli", "_stage", stage, "--id", video_id]
    try:
        completed = subprocess.run(
            command, timeout=_timeout_for(cfg, stage), cwd=str(cfg.root),
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        log.error("stage.timeout", video_id=video_id, stage=stage,
                  timeout_s=_timeout_for(cfg, stage))
        return False
    if completed.returncode != 0:
        log.error("stage.subprocess_failed", video_id=video_id, stage=stage,
                  returncode=completed.returncode, stderr=completed.stderr[-800:])
        return False
    return True


def discover_all(cfg: Config, conn) -> int:
    """Poll every enabled channel and enqueue anything new."""
    import re as _re

    from .sources import youtube

    added = 0
    backfill = int(cfg.get("discover", "backfill_days", 7))
    for channel in conn.execute("SELECT * FROM channels WHERE enabled=1"):
        try:
            refs = youtube.discover(channel["id"], backfill)
        except Exception as exc:
            log.error("discover.failed", channel_id=channel["id"], error=str(exc)[:200])
            continue
        for ref in refs:
            if channel["title_include"] and not _re.search(channel["title_include"], ref.title):
                continue
            if channel["title_exclude"] and _re.search(channel["title_exclude"], ref.title):
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO videos (id, channel_id, title, published_at, "
                "discovered_at, status) VALUES (?,?,?,?,?,?)",
                (ref.id, channel["id"], ref.title, ref.published_at, D.now_iso(), D.NEW),
            )
            if cur.rowcount:
                added += 1
                log.info("discover.new", video_id=ref.id, title=ref.title[:80])
    return added


def run_once(cfg: Config, conn, limit: int | None = None) -> dict:
    """Drive the queue to completion, sequentially, oldest first."""
    limit = limit or int(cfg.get("runner", "max_videos_per_run", 12))
    # Reap stages killed by a previous run before claiming any work, or a video
    # whose process was killed re-runs at full cost forever with no attempt
    # counter, no backoff and no path to abandonment.
    longest = max(
        int(cfg.get("runner", f"timeout_{s}_s", 900)) for s in STAGE_FUNCS
    )
    reaped = db_reap(conn, longest)
    if reaped:
        log.warning("run.reaped_orphan_stages", count=reaped)
    stats = {"processed": 0, "completed": 0, "failed": 0}
    for video in D.claim_queue(conn, limit):
        video_id = video["id"]
        stats["processed"] += 1
        while True:
            stage = D.next_stage_for(conn, video_id)
            if stage is None:
                break
            if not run_stage_subprocess(cfg, video_id, stage):
                stats["failed"] += 1
                break
        row = conn.execute("SELECT status FROM videos WHERE id=?", (video_id,)).fetchone()
        if row and row["status"] == D.DONE:
            stats["completed"] += 1
    return stats
