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
from datetime import UTC, datetime
from pathlib import Path

from . import db as D
from .db import reap_orphan_runs as db_reap
from .config import Config
from .logging import get_logger
from .summarize import in_focus

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
    content = segments

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
                dominant_lang(content), len(content),
                _mean_confidence(content), D.now_iso(),
            ),
        )

    out = cfg.data_dir / "normalized" / f"{video['id']}.json"
    write_normalized(out, video["id"], content, ledger)
    log.info("normalize.ledger", video_id=video["id"], entries=len(ledger))
    return out


def _mean_confidence(segments) -> float | None:
    values = [s.confidence for s in segments if s.confidence is not None]
    return sum(values) / len(values) if values else None


def _identify_speakers(cfg: Config, conn, video: dict,
                       raw_segments: list, wav) -> dict[float, str] | None:
    """Voice-identify this video's segments, or return None if we cannot.

    Never fatal. Speaker identification improves a summary; it is not required
    to produce one, and a network failure or a machine without torch installed
    must not cost the summary itself. Returning None makes the prompt fall back
    to refusing all attribution, which is the safe direction.

    `wav` is downloaded once by the caller and shared with the cross-check --
    a 2.5-hour show is ~570 MB, and fetching it twice would double the cost of
    this stage for no benefit.
    """
    if not cfg.get("voice", "enabled", True):
        return None
    try:
        from .summarize import speaker_map
        from .voice import (
            DEFAULT_MARGIN, DEFAULT_THRESHOLD, identify,
            store_attributions, voiceprints,
        )
    except ImportError:
        return None

    existing = speaker_map(conn, video["id"])
    if existing is not None:                    # already identified; don't re-pay
        return existing
    if not voiceprints(conn):
        log.info("voice.skipped", video_id=video["id"], reason="no voiceprints")
        return None

    try:
        rows = identify(
            conn, wav, raw_segments,
            float(cfg.get("voice", "threshold", DEFAULT_THRESHOLD)),
            float(cfg.get("voice", "margin", DEFAULT_MARGIN)),
        )
        named = store_attributions(conn, video["id"], rows)
        log.info("voice.identified", video_id=video["id"],
                 attributed=named, segments=len(rows))
        return speaker_map(conn, video["id"])
    except Exception as exc:                    # noqa: BLE001 - never fatal
        log.warning("voice.failed", video_id=video["id"], error=str(exc)[:200])
        return None


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


def _crosscheck_figures(cfg: Config, conn, video: dict, wav) -> dict[str, int]:
    """Give every figure a second reading. Never fatal.

    Runs after voice ID inside the same stage, sharing its wav -- audio is the
    expensive part and it is already on disk. Sequentially, never alongside:
    ASR peaks at 5.2 GB on a machine that is already swapping.
    """
    from .crosscheck import ABSENT, UNCHECKED, resolve_window, spans_for, values_in
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

    heard_by_span = {(s.start, s.end): values_in(s.text) for s in segments}

    # Group ledger rows by the merged span that covers their timestamp, and
    # judge each span's figures together via resolve_window rather than one
    # row against everything heard nearby. `spans_for` merges overlapping
    # windows, so a dense passage becomes one span holding several unrelated
    # figures -- comparing each in isolation let correct figures dispute
    # against other sentences' numbers (measured: 25% precision on a dense
    # passage in MgN00MCDDRM).
    rows_by_span: dict[tuple[float, float], list] = {span: [] for span in spans}
    uncovered = []
    for row in rows:
        for span in spans:
            if span[0] <= row["start_s"] <= span[1]:
                rows_by_span[span].append(row)
                break
        else:
            uncovered.append(row)

    counts: dict[str, int] = {}

    def _write(row, state, rival):
        counts[state] = counts.get(state, 0) + 1
        # _fmt, not repr: the ledger's own `normalized` is written by _fmt,
        # which renders whole numbers as "2900000000". repr() would write
        # "29900000000.0", and Task 7 compares the two as strings.
        conn.execute(
            "UPDATE number_ledger SET crosscheck=?, asr_normalized=?, asr_model=? "
            "WHERE id=?",
            (state, _fmt(rival), engine.id, row["id"]),
        )

    with D.transaction(conn):
        for span, span_rows in rows_by_span.items():
            if not span_rows:
                continue
            captions = []
            for row in span_rows:
                try:
                    captions.append(float(row["normalized"]) if row["normalized"] else None)
                except (TypeError, ValueError):
                    captions.append(None)
            verdicts = resolve_window(captions, heard_by_span.get(span, []))
            for row, (state, rival) in zip(span_rows, verdicts):
                _write(row, state, rival)

        # A row whose timestamp fell in no span at all (spans_for can drop a
        # window that clamps to zero width at a video's edge) was never given
        # to ASR, so it is unchecked/absent rather than judged.
        for row in uncovered:
            try:
                caption = float(row["normalized"]) if row["normalized"] else None
            except (TypeError, ValueError):
                caption = None
            _write(row, UNCHECKED if caption is None else ABSENT, None)

    log.info("crosscheck.done", video_id=video["id"], **counts)
    return counts


def stage_identify(cfg: Config, conn, video: dict) -> None:
    """Voice-identify this video's segments, then cross-check its figures.

    Its own stage, deliberately. This ran inside `stage_summarize` at first,
    and that was wrong twice over. It loads torch and embeds an hour of audio,
    so summarize went from minutes to an average of eleven and a maximum of
    fifty-nine against a sixty-minute timeout -- five videos died mid-stage as
    a result. And the architecture already says why each stage is its own
    subprocess: heavy model memory is not reliably returned to the OS within a
    process, which matters on a 16 GB machine that is already swapping.

    Both steps are never fatal. A video with no attribution still summarises;
    it just attributes nobody, which is the safe direction. A video with no
    cross-check still publishes; its figures are simply left unchecked.

    The wav is downloaded once here and shared between the two steps, which
    run strictly one after the other -- ASR alone peaks at 5.2 GB, so it must
    never run alongside voice identification's own model.
    """
    if not cfg.get("voice", "enabled", True) and not cfg.get("asr", "crosscheck", True):
        return None
    artifact = D.get_artifact(conn, video["id"], "normalized")
    if artifact is None:
        raise D.StageError("no normalized artifact", D.RETRYABLE)
    segments = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))["segments"]

    from .voice import audio_for

    try:
        with audio_for(video["id"], cfg.data_dir / "audio") as wav:
            _identify_speakers(cfg, conn, video, segments, wav=wav)
            _crosscheck_figures(cfg, conn, video, wav)
    except Exception as exc:                    # noqa: BLE001 - never fatal
        log.warning("audio_stage.failed", video_id=video["id"], error=str(exc)[:200])
    return None


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

    # Identification is its own stage now; here we only read what it stored.
    from .summarize import speaker_map
    speakers = speaker_map(conn, video["id"])

    transcript_artifact = D.get_artifact(conn, video["id"], "transcript")
    payload, state, checks = summarize(
        cfg, segments, ledger, data.get("dominant_lang", "yue"), log,
        title=video.get("title", ""), speakers=speakers,
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


def _is_newsworthy(cfg: Config, video: dict) -> bool:
    """Whether this video is recent enough to notify about.

    A digest is a notification, and a notification about something published
    last December is noise. Backfilling a channel's back catalogue queues
    hundreds of videos through the same pipeline as today's upload, and without
    this every one of them would arrive as a separate Telegram message.

    The markdown is always written — only the notification is suppressed — so
    nothing is lost, it simply does not interrupt.
    """
    days = int(cfg.get("publish", "notify_within_days", 3))
    if days <= 0:
        return True
    published = video.get("published_at")
    if not published:
        return True                       # unknown age: treat as current
    try:
        when = datetime.fromisoformat(published)
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (datetime.now(UTC) - when).days <= days


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

    if cfg.get("publish", "telegram", False) and _is_newsworthy(cfg, video):
        send_telegram(cfg, render_telegram(view, data["payload"], checks))

    from .summarize import hosts_from_title
    from .views import parse_views, store_views, sync_instruments

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
        summary_id = conn.execute(
            "SELECT id FROM summaries WHERE video_id=? ORDER BY id DESC LIMIT 1",
            (video["id"],),
        ).fetchone()["id"]

    # Market views are extracted here rather than at summarise time so that a
    # prompt change can be replayed over existing summaries without re-paying
    # for generation.
    sync_instruments(conn)
    views = parse_views(data["payload"], hosts_from_title(video.get("title", "")))
    if views:
        n = store_views(conn, dict(video), views, summary_id,
                        data.get("prompt_version", "?"))
        log.info("publish.views", video_id=video["id"], views=n)
    return path


STAGE_FUNCS = {
    "fetch": stage_fetch,
    "transcribe": stage_transcribe,
    "normalize": stage_normalize,
    "identify": stage_identify,
    "summarize": stage_summarize,
    "publish": stage_publish,
}

ARTIFACT_KIND = {
    "fetch": "audio",
    "transcribe": "transcript",
    "normalize": "normalized",
    "identify": None,
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
    stats = {"processed": 0, "completed": 0, "failed": 0, "skipped": 0}
    for video in D.claim_queue(conn, limit):
        video_id = video["id"]
        # Checked here rather than at discovery: an episode's parts carry no
        # host names, so a part discovered before its parent cannot be judged
        # yet. Re-evaluating each run means it settles once the parent arrives,
        # and costs nothing until then.
        keep, reason = in_focus(conn, dict(video))
        if not keep:
            with D.transaction(conn):
                conn.execute("UPDATE videos SET status = ? WHERE id = ?",
                             (D.SKIPPED, video_id))
            log.info("run.skipped", video_id=video_id, reason=reason,
                     title=video["title"][:80])
            stats["skipped"] += 1
            continue
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
