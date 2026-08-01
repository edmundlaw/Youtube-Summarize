"""Command line entry point.

`cli.py` orchestrates; it never does heavy work in-process. Each pipeline stage
runs as a separate subprocess (`ytdigest _stage <name> --id <video_id>`) so that
the ASR model's unified memory is genuinely returned to the OS before the
summariser starts. On a 16 GB machine that is not a nicety.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from . import db as D
from .config import REPO_ROOT, load_config
from .logging import get_logger, setup_logging

MIGRATIONS = REPO_ROOT / "migrations"


def _open() -> tuple:
    cfg = load_config()
    cfg.ensure_dirs()
    conn = D.open_db(cfg.db_path, MIGRATIONS)
    return cfg, conn


@click.group()
@click.option("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR")
@click.pass_context
def main(ctx: click.Context, log_level: str) -> None:
    """ytdigest — YouTube finance video digests with verified numbers."""
    cfg = load_config()
    setup_logging(cfg.log_path, log_level)
    ctx.ensure_object(dict)


# --- status ----------------------------------------------------------------


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def status(as_json: bool) -> None:
    """Queue counts, oldest pending item, last successful run, anything abandoned."""
    cfg, conn = _open()
    counts = D.status_counts(conn)
    oldest = D.oldest_pending(conn)
    abandoned = D.abandoned_items(conn)
    last_run = D.last_successful_run(conn)
    timings = D.stage_timings(conn)
    channels = conn.execute(
        "SELECT COUNT(*) AS n, SUM(enabled) AS on_ FROM channels"
    ).fetchone()

    stale_hours = cfg.get("publish", "stale_queue_hours", 48)
    stale = False
    if oldest is not None:
        discovered = datetime.fromisoformat(oldest["discovered_at"])
        stale = datetime.now(UTC) - discovered > timedelta(hours=stale_hours)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "channels": {"total": channels["n"], "enabled": channels["on_"] or 0},
                    "counts": counts,
                    "oldest_pending": dict(oldest) if oldest else None,
                    "queue_stale": stale,
                    "last_successful_publish": last_run,
                    "abandoned": [dict(r) for r in abandoned],
                    "stage_timings": [dict(r) for r in timings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    click.echo(f"channels        {channels['n']} ({channels['on_'] or 0} enabled)")
    click.echo(f"database        {cfg.db_path}")
    if not counts:
        click.echo("queue           empty")
    else:
        click.echo("queue")
        for name in (
            D.NEW, D.FETCHED, D.TRANSCRIBED, D.NORMALIZED,
            D.SUMMARIZED, D.DONE, D.FAILED, D.ABANDONED,
        ):
            if name in counts:
                click.echo(f"  {name:<14} {counts[name]}")

    if oldest is not None:
        marker = "  ** STALE **" if stale else ""
        click.echo(
            f"oldest pending  [{oldest['status']}] {oldest['id']} "
            f"discovered {oldest['discovered_at']}{marker}"
        )
    click.echo(f"last publish    {last_run or 'never'}")

    if timings:
        click.echo("stage timings   (successful runs)")
        for row in timings:
            click.echo(
                f"  {row['stage']:<12} n={row['runs']:<4} "
                f"avg={row['avg_ms'] / 1000:.1f}s  max={row['max_ms'] / 1000:.1f}s"
            )

    if abandoned:
        click.echo(f"\nabandoned       {len(abandoned)} — these never auto-retry:")
        for row in abandoned:
            click.echo(
                f"  {row['id']}  {row['stage']:<10} [{row['error_class']}] "
                f"{(row['error_text'] or '')[:70]}"
            )
        click.echo("  retry one with:  ytdigest retry <video_id>")


# --- channels --------------------------------------------------------------


@main.group()
def channel() -> None:
    """Manage watched channels."""


@channel.command("add")
@click.argument("url")
@click.option("--lang", default=None, help="Language hint: yue | zh | en. Default auto.")
@click.option("--min-duration", type=int, default=180, help="Skip anything shorter (s).")
@click.option("--max-duration", type=int, default=10800, help="Skip anything longer (s).")
@click.option("--include", default=None, help="Only titles matching this regex.")
@click.option("--exclude", default=None, help="Skip titles matching this regex.")
def channel_add(
    url: str,
    lang: str | None,
    min_duration: int,
    max_duration: int,
    include: str | None,
    exclude: str | None,
) -> None:
    """Resolve a channel URL/handle to its UC... id and start watching it."""
    from .sources.youtube import resolve_channel

    cfg, conn = _open()
    log = get_logger()
    try:
        channel_id, title = resolve_channel(url)
    except Exception as exc:
        raise click.ClickException(f"could not resolve {url}: {exc}") from exc

    conn.execute(
        "INSERT INTO channels (id, title, lang_hint, enabled, min_duration_s, "
        "max_duration_s, title_include, title_exclude, added_at) "
        "VALUES (?,?,?,1,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET title=excluded.title, lang_hint=excluded.lang_hint, "
        "  enabled=1, min_duration_s=excluded.min_duration_s, "
        "  max_duration_s=excluded.max_duration_s, title_include=excluded.title_include, "
        "  title_exclude=excluded.title_exclude",
        (channel_id, title, lang, min_duration, max_duration, include, exclude, D.now_iso()),
    )
    log.info("channel.added", channel_id=channel_id, title=title, lang_hint=lang)
    click.echo(f"added {channel_id}  {title}")


@channel.command("list")
def channel_list() -> None:
    """Show watched channels."""
    _, conn = _open()
    rows = list(conn.execute("SELECT * FROM channels ORDER BY title"))
    if not rows:
        click.echo("no channels yet — add one with: ytdigest channel add <url>")
        return
    for row in rows:
        state = "on " if row["enabled"] else "off"
        lang = row["lang_hint"] or "auto"
        click.echo(f"[{state}] {row['id']}  {row['title']}  lang={lang}")


@channel.command("disable")
@click.argument("channel_id")
def channel_disable(channel_id: str) -> None:
    """Stop watching a channel without deleting its history."""
    _, conn = _open()
    conn.execute("UPDATE channels SET enabled=0 WHERE id=?", (channel_id,))
    click.echo(f"disabled {channel_id}")


# --- doctor ----------------------------------------------------------------


@main.command()
@click.option("--limit", type=int, default=None, help="Max videos this run.")
@click.option("--no-discover", is_flag=True, help="Skip RSS polling.")
def run(limit: int | None, no_discover: bool) -> None:
    """Poll channels and drive the queue. This is what launchd calls."""
    from .runner import discover_all, run_once, single_instance

    cfg, conn = _open()
    log = get_logger()
    with single_instance(cfg.lock_path) as acquired:
        if not acquired:
            # launchd firing while a long transcription runs must be a no-op.
            log.info("run.skipped", reason="another instance holds the lock")
            click.echo("another run is in progress — nothing to do")
            return
        if not no_discover:
            found = discover_all(cfg, conn)
            click.echo(f"discovered {found} new video(s)")
        stats = run_once(cfg, conn, limit)
        # Prices and grading run every time so the record self-heals: Yahoo
        # intermittently refuses connections from this machine, and a view that
        # could not be graded today must be picked up automatically once it can.
        try:
            from .prices import default_start, fetch_prices, symbols_needed
            from .resolver import resolve_all

            wanted = symbols_needed(conn)
            if wanted:
                got = fetch_prices(conn, wanted, default_start(),
                                   datetime.now(UTC).date().isoformat())
                missing = [s for s, n in got.items() if n == 0]
                log.info("run.prices", fetched=sum(got.values()),
                         no_data=len(missing))
            log.info("run.resolved", **resolve_all(conn))
        except Exception as exc:
            # Never fail a publishing run because grading could not happen.
            log.warning("run.grading_skipped", error=str(exc)[:200])
        click.echo(
            f"processed {stats['processed']}, completed {stats['completed']}, "
            f"failed {stats['failed']}"
        )
        _notify_if_broken(cfg, conn)


def _notify_if_broken(cfg, conn) -> None:
    """Silence must not be ambiguous between 'nothing new' and 'broken since
    Tuesday'."""
    from .publish import notify

    abandoned = D.abandoned_items(conn)
    oldest = D.oldest_pending(conn)
    stale_hours = int(cfg.get("publish", "stale_queue_hours", 48))
    stale = False
    if oldest is not None:
        age = datetime.now(UTC) - datetime.fromisoformat(oldest["discovered_at"])
        stale = age > timedelta(hours=stale_hours)
    if not abandoned and not stale:
        return
    lines = ["⚠️ <b>ytdigest 需要注意</b>"]
    if abandoned:
        lines.append(f"{len(abandoned)} 條片放棄咗（唔會自動重試）：")
        lines += [f"  • {r['id']} {r['stage']} — {(r['error_text'] or '')[:80]}"
                  for r in abandoned[:5]]
    if stale:
        lines.append(f"最舊未處理項目已經超過 {stale_hours} 小時。")
    notify(cfg, "\n".join(lines))


@main.command("_stage", hidden=True)
@click.argument("stage")
@click.option("--id", "video_id", required=True)
def _stage(stage: str, video_id: str) -> None:
    """Run a single stage in this process. Spawned by the runner."""
    from .runner import run_stage_inprocess

    cfg, conn = _open()
    try:
        run_stage_inprocess(cfg, conn, video_id, stage)
    except Exception:
        sys.exit(1)


@main.command()
@click.argument("video_id")
def retry(video_id: str) -> None:
    """Clear the abandoned state on one video so it re-enters the queue."""
    _, conn = _open()
    row = conn.execute("SELECT status FROM videos WHERE id=?", (video_id,)).fetchone()
    if row is None:
        raise click.ClickException(f"unknown video {video_id}")
    with D.transaction(conn):
        conn.execute(
            "UPDATE stage_runs SET next_retry_at=NULL WHERE video_id=? AND status='failed'",
            (video_id,),
        )
        conn.execute("UPDATE videos SET status=? WHERE id=?", (D.FAILED, video_id))
    click.echo(f"{video_id}: {row['status']} -> failed (will retry on next run)")


@main.command()
@click.argument("url_or_id")
def add(url_or_id: str) -> None:
    """Enqueue a single video directly, bypassing channel discovery."""
    import re as _re

    from .sources import youtube

    cfg, conn = _open()
    match = _re.search(r"(?:v=|/live/|youtu\.be/)([\w-]{11})", url_or_id)
    video_id = match.group(1) if match else url_or_id
    info = youtube.probe(video_id)
    channel_id = info.get("channel_id")
    conn.execute(
        "INSERT OR IGNORE INTO channels (id, title, enabled, added_at) VALUES (?,?,1,?)",
        (channel_id, info.get("channel") or channel_id, D.now_iso()),
    )
    published = info.get("upload_date") or ""
    published_iso = (
        f"{published[:4]}-{published[4:6]}-{published[6:8]}T00:00:00+00:00"
        if len(published) == 8 else D.now_iso()
    )
    conn.execute(
        "INSERT OR REPLACE INTO videos (id, channel_id, title, published_at, "
        "duration_s, discovered_at, status) VALUES (?,?,?,?,?,?,?)",
        (video_id, channel_id, info.get("title", video_id), published_iso,
         info.get("duration"), D.now_iso(), D.NEW),
    )
    click.echo(f"queued {video_id}  {info.get('title', '')[:70]}")


@main.command("views")
@click.option("--instrument", default=None, help="Filter by symbol, e.g. ^HSI or 0700.HK")
@click.option("--speaker", default=None, help="Filter by speaker (substring).")
@click.option("--direction", default=None, help="long | short | neutral | avoid | exit")
@click.option("--since", default=None, help="ISO date, e.g. 2026-07-01")
@click.option("--verified-only", is_flag=True, help="Only levels matched to the ledger.")
@click.option("--unmapped", is_flag=True, help="Show views whose instrument did not map.")
@click.option("--csv", "as_csv", is_flag=True, help="CSV to stdout, for charting/backtest.")
@click.option("--limit", type=int, default=60)
def views_cmd(instrument, speaker, direction, since, verified_only, unmapped,
              as_csv, limit):
    """Query the market views extracted from videos."""
    import csv as _csv

    _, conn = _open()
    where, args = ["1=1"], []
    if instrument:
        where.append("v.instrument = ?"); args.append(instrument)
    if speaker:
        where.append("v.speaker LIKE ?"); args.append(f"%{speaker}%")
    if direction:
        where.append("v.direction = ?"); args.append(direction)
    if since:
        where.append("v.stated_at >= ?"); args.append(since)
    if verified_only:
        where.append("v.level_verified = 1")
    if unmapped:
        where.append("v.instrument IS NULL")

    rows = list(conn.execute(
        f"""SELECT v.stated_at, v.speaker, v.instrument, v.instrument_raw,
                   v.asset_class, v.direction, v.conviction, v.level_type,
                   v.level_value, v.level_unit, v.level_verified, v.horizon,
                   v.horizon_ends_at, v.outcome, v.thesis, v.video_id, v.start_s,
                   v.entry_basis, v.condition, v.stance,
                   c.title AS channel
            FROM views v JOIN channels c ON c.id = v.channel_id
            WHERE {' AND '.join(where)}
            ORDER BY v.stated_at DESC LIMIT ?""", (*args, limit)))

    if as_csv:
        writer = _csv.writer(sys.stdout)
        writer.writerow([
            "stated_at", "speaker", "instrument", "instrument_raw", "asset_class",
            "direction", "conviction", "level_type", "level_value", "level_unit",
            "level_verified", "horizon", "entry_basis", "condition", "stance",
            "horizon_ends_at", "outcome", "video_id", "start_s", "url", "thesis",
        ])
        for r in rows:
            writer.writerow([
                r["stated_at"], r["speaker"] or "", r["instrument"] or "",
                r["instrument_raw"], r["asset_class"] or "", r["direction"],
                r["conviction"] or "", r["level_type"] or "",
                r["level_value"] if r["level_value"] is not None else "",
                r["level_unit"] or "", r["level_verified"], r["horizon"] or "",
                r["entry_basis"] or "", r["condition"] or "", r["stance"] or "",
                r["horizon_ends_at"] or "", r["outcome"] or "",
                r["video_id"], int(r["start_s"]),
                f"https://youtu.be/{r['video_id']}?t={int(r['start_s'])}",
                r["thesis"],
            ])
        return

    if not rows:
        click.echo("no views match."); return
    for r in rows:
        level = ""
        if r["level_value"] is not None:
            mark = "" if r["level_verified"] else " (unverified)"
            level = f"  {r['level_type'] or 'level'} {r['level_value']:g}{mark}"
        sym = r["instrument"] or f"?{r['instrument_raw']}"
        # A conditional call must never read as an immediate one.
        trigger = ""
        if r["entry_basis"] and r["entry_basis"] != "unspecified":
            trigger = "  [" + {"on_rally": "只在反彈時", "on_dip": "只在回落時",
                               "on_break": "只在破位時",
                               "on_confirmation": "待確認"}.get(
                                   r["entry_basis"], r["entry_basis"]) + "]"
        click.echo(
            f"{r['stated_at'][:10]}  {sym:<10} {r['direction']:<8}"
            f"{level:<26}{trigger:<14}{(r['speaker'] or '—')[:14]:<16}"
            f"https://youtu.be/{r['video_id']}?t={int(r['start_s'])}"
        )
        click.echo(f"           {r['thesis'][:110]}")
    click.echo(f"\n{len(rows)} views. Add --csv to export.")


@main.command("views-reindex")
def views_reindex() -> None:
    """Re-extract views from stored summaries, without calling the model.

    Extraction rules (instrument aliases, speaker rosters, level verification)
    keep improving. Re-running them over summaries already paid for is free;
    regenerating the summaries is not.
    """
    import json as _json

    from .summarize import hosts_from_title
    from .views import parse_views, store_views, sync_instruments

    cfg, conn = _open()
    sync_instruments(conn)
    total = 0
    for video in conn.execute("SELECT * FROM videos WHERE status = ?", (D.DONE,)):
        path = cfg.data_dir / "normalized" / f"{video['id']}.summary.json"
        if not path.exists():
            continue
        data = _json.loads(path.read_text(encoding="utf-8"))
        summary = conn.execute(
            "SELECT id FROM summaries WHERE video_id=? ORDER BY id DESC LIMIT 1",
            (video["id"],),
        ).fetchone()
        views = parse_views(data.get("payload", {}),
                            hosts_from_title(video["title"]))
        if views:
            total += store_views(conn, dict(video), views,
                                 summary["id"] if summary else None,
                                 data.get("prompt_version", "?"))
    click.echo(f"reindexed {total} views")


@main.command("prices")
@click.option("--start", default=None, help="ISO date. Default: last calendar year.")
@click.option("--symbol", multiple=True, help="Limit to these symbols.")
def prices_cmd(start, symbol):
    """Fetch daily price bars for the instruments views refer to."""
    from datetime import date

    from .prices import coverage, default_start, fetch_prices, symbols_needed

    _, conn = _open()
    wanted = list(symbol) or symbols_needed(conn)
    if not wanted:
        click.echo("no mapped instruments yet — run the pipeline first."); return
    click.echo(f"fetching {len(wanted)} symbols from {start or default_start()}...")
    stored = fetch_prices(conn, wanted, start or default_start(),
                          date.today().isoformat())
    empty = [s for s, n in stored.items() if n == 0]
    for row in coverage(conn):
        click.echo(f"  {row['symbol']:<12} {row['bars']:>5} bars  "
                   f"{row['first']} .. {row['last']}")
    if empty:
        # Never silent: a symbol with no data becomes an ungraded view later,
        # and without this line there would be nothing explaining why.
        click.echo(f"\nNO DATA for: {', '.join(empty)}")
        click.echo("Those instruments cannot be graded until a source is found.")


@main.command("scorecard")
@click.option("--resolve/--no-resolve", default=True, help="Grade pending views first.")
@click.option("--min-graded", type=int, default=5,
              help="Below this many graded calls, show no hit rate.")
def scorecard_cmd(resolve: bool, min_graded: int) -> None:
    """Speaker track records, with everything that could not be graded shown."""
    from .resolver import resolve_all, scorecard

    _, conn = _open()
    if resolve:
        counts = resolve_all(conn)
        click.echo("graded: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        click.echo("")

    rows = scorecard(conn, min_graded)
    click.echo(f"{'speaker':<20}{'hit':>5}{'miss':>6}{'rate':>8}"
               f"{'void':>7}{'unresolv':>10}{'pending':>9}{'total':>7}"
               f"{'voice':>7}{'guess':>7}")
    click.echo("-" * 86)
    for r in rows:
        rate = f"{r['hit_rate']*100:.0f}%" if r["hit_rate"] is not None else "—"
        click.echo(
            f"{r['speaker'][:19]:<20}{r['hit']:>5}{r['missed']:>6}{rate:>8}"
            f"{r['void']:>7}{r['unresolvable']:>10}{r['pending']:>9}{r['total']:>7}"
            f"{r['by_voice']:>7}{r['by_guess']:>7}"
        )
    click.echo("")
    click.echo("voice = name confirmed by voiceprint. guess = the model's")
    click.echo("       inference from unlabelled captions, before voice ID existed.")
    click.echo("       Only 'voice' counts are trustworthy for judging a person.")
    click.echo("rate is over graded calls only (hit+miss). '—' means too few to judge.")
    click.echo("void = conditional call whose trigger never fired — the speaker")
    click.echo("       never advised acting, so it counts neither way.")
    click.echo("unresolvable = no horizon, no level, unverified level, or unmapped")
    click.echo("       instrument. Not a failure of the speaker.")


@main.command("enroll")
@click.argument("speaker")
@click.option("--video", "videos", multiple=True, required=True,
              help="Video IDs where SPEAKER is the only voice. Repeatable.")
def enroll_cmd(speaker: str, videos: tuple[str, ...]) -> None:
    """Build a voiceprint from videos where only SPEAKER talks.

    Use solo videos only. A voiceprint averaged over someone else's audio will
    quietly attribute their calls to SPEAKER, which is the exact failure this
    whole subsystem exists to prevent.
    """
    from .views import canonical_speaker
    from .voice import audio_for, enroll, purity_warning

    cfg, conn = _open()
    canonical = canonical_speaker(speaker) or speaker
    if canonical != speaker:
        click.echo(f"canonicalised '{speaker}' -> '{canonical}'")

    audio_dir = cfg.data_dir / "audio"
    paths, keep = [], []
    try:
        for video_id in videos:
            click.echo(f"downloading {video_id}...")
            manager = audio_for(video_id, audio_dir)
            path = manager.__enter__()
            keep.append(manager)
            paths.append(path)
            click.echo(f"  {path.stat().st_size / 1e6:.0f} MB")
        click.echo("embedding (first run downloads the model, ~30s)...")
        result = enroll(conn, canonical, paths, source_note=",".join(videos))
    finally:
        for manager in keep:                      # deletes every wav
            manager.__exit__(None, None, None)

    click.echo(f"enrolled {result['speaker']}: {result['windows']} windows, "
               f"{result['seconds'] / 60:.0f} min of voice, "
               f"purity {result['purity']:.0%}. Audio deleted.")
    warning = purity_warning(result)
    if warning:
        click.echo("")
        click.secho(f"WARNING: {warning}", fg="yellow")
        click.echo("Delete it with:  ytdigest unenroll "
                   f"'{result['speaker']}'")


@main.command("identify")
@click.argument("video_id")
@click.option("--threshold", type=float, default=None)
@click.option("--margin", type=float, default=None)
def identify_cmd(video_id: str, threshold: float | None, margin: float | None) -> None:
    """Attribute a video's caption segments to enrolled speakers."""
    import json as _json

    from .voice import (
        DEFAULT_MARGIN, DEFAULT_THRESHOLD, audio_for, identify,
        store_attributions, voiceprints,
    )

    cfg, conn = _open()
    if not voiceprints(conn):
        click.echo("no voiceprints yet — run `ytdigest enroll` first."); return

    path = cfg.data_dir / "normalized" / f"{video_id}.json"
    if not path.exists():
        click.echo(f"no transcript for {video_id}"); return
    segments = _json.loads(path.read_text(encoding="utf-8"))["segments"]

    with audio_for(video_id, cfg.data_dir / "audio") as wav:
        rows = identify(conn, wav, segments,
                        threshold if threshold is not None else DEFAULT_THRESHOLD,
                        margin if margin is not None else DEFAULT_MARGIN)
    named = store_attributions(conn, video_id, rows)

    tally: dict[str, int] = {}
    for row in rows:
        tally[row.speaker or "(unattributed)"] = tally.get(row.speaker or "(unattributed)", 0) + 1
    click.echo(f"{named}/{len(rows)} segments attributed. Audio deleted.")
    for name, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        click.echo(f"  {name:<24} {count:>4}")


@main.command("unenroll")
@click.argument("speaker")
def unenroll_cmd(speaker: str) -> None:
    """Delete a voiceprint. Segments already attributed with it are cleared too,
    since they were decided by a voiceprint no longer trusted."""
    from .views import canonical_speaker

    _, conn = _open()
    canonical = canonical_speaker(speaker) or speaker
    with D.transaction(conn):
        cleared = conn.execute(
            "UPDATE segment_speakers SET speaker=NULL WHERE speaker=?",
            (canonical,)).rowcount
        conn.execute(
            "UPDATE views SET speaker=NULL, attribution='none' "
            "WHERE speaker=? AND attribution='voice'", (canonical,))
        gone = conn.execute("DELETE FROM voiceprints WHERE speaker=?",
                            (canonical,)).rowcount
    click.echo(f"removed {gone} voiceprint(s) for {canonical}; "
               f"cleared {cleared} segment attributions.")


@main.command("voices")
def voices_cmd() -> None:
    """Enrolled voiceprints."""
    _, conn = _open()
    rows = list(conn.execute(
        "SELECT speaker, n_clips, total_s, source_note, updated_at "
        "FROM voiceprints ORDER BY speaker"))
    if not rows:
        click.echo("none enrolled. `ytdigest enroll <name> --video <id>`"); return
    for r in rows:
        click.echo(f"{r['speaker']:<22} {r['n_clips']:>4} windows  "
                   f"{r['total_s'] / 60:>5.0f} min  from {r['source_note']}")


@main.command("telegram-setup")
@click.option("--token", default=None, help="Bot token from @BotFather. Reads .env if omitted.")
@click.option("--wait", default=120, help="Seconds to wait for your message.")
def telegram_setup(token: str | None, wait: int) -> None:
    """Discover the chat id for a new bot, then write it to .env.

    Telegram has no API to create a bot — @BotFather is a human-only chat. But
    once the bot exists, everything after that is automatable: add it to the
    destination, send it any message, and this command reads the chat id off
    getUpdates so you never have to look it up by hand.
    """
    import time

    import httpx

    cfg = load_config()
    token = token or cfg.secret("TELEGRAM_BOT_TOKEN")
    if not token:
        raise click.ClickException(
            "No token. Create a bot first:\n"
            "  1. Telegram -> @BotFather -> /newbot\n"
            "  2. Pick a display name, then a username ending in 'bot'\n"
            "  3. Re-run:  ytdigest telegram-setup --token <token>"
        )

    api = f"https://api.telegram.org/bot{token}"
    try:
        me = httpx.get(f"{api}/getMe", timeout=20).json()
    except Exception as exc:
        raise click.ClickException(f"could not reach Telegram: {exc}") from exc
    if not me.get("ok"):
        raise click.ClickException(f"token rejected: {me.get('description')}")
    bot = me["result"]
    click.echo(f"bot @{bot['username']} ({bot['first_name']}) verified.")

    # Drain anything already queued so a stale message can't be mistaken for
    # the user's fresh one.
    offset = 0
    seen = httpx.get(f"{api}/getUpdates", timeout=20).json().get("result", [])
    if seen:
        offset = seen[-1]["update_id"] + 1

    click.echo(
        "\nNow, in Telegram:\n"
        f"  1. Add @{bot['username']} to the group/channel you want digests in\n"
        "     (or just open a direct chat with it)\n"
        "  2. Send any message there — 'hello' is fine\n"
        f"\nwaiting up to {wait}s..."
    )

    deadline = time.time() + wait
    chat = None
    while time.time() < deadline and chat is None:
        data = httpx.get(
            f"{api}/getUpdates", params={"offset": offset, "timeout": 10}, timeout=25
        ).json()
        for update in data.get("result", []):
            offset = update["update_id"] + 1
            payload = (
                update.get("message")
                or update.get("channel_post")
                or update.get("my_chat_member")
            )
            if payload and payload.get("chat"):
                chat = payload["chat"]
                thread = (payload.get("message_thread_id") if payload else None)
                break
        else:
            continue
        break

    if chat is None:
        raise click.ClickException(
            "No message seen. If you added the bot to a group, Telegram hides "
            "group messages from bots by default — either send a message that "
            "@mentions the bot, or in @BotFather use /setprivacy -> Disable."
        )

    title = chat.get("title") or chat.get("username") or chat.get("first_name")
    click.echo(f"\nfound chat: {title!r}  type={chat['type']}  id={chat['id']}")

    env_path = cfg.root / ".env"
    keep = [
        line
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip("# ").startswith("TELEGRAM_")
    ]
    keep += [f"TELEGRAM_BOT_TOKEN={token}", f"TELEGRAM_CHAT_ID={chat['id']}"]
    if thread:
        keep.append(f"TELEGRAM_MESSAGE_THREAD_ID={thread}")
    env_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    env_path.chmod(0o600)
    click.echo(f"wrote TELEGRAM_* to {env_path} (chmod 600)")

    confirm = httpx.post(
        f"{api}/sendMessage",
        json={"chat_id": chat["id"], "text": "ytdigest is connected. Digests will arrive here."}
        | ({"message_thread_id": thread} if thread else {}),
        timeout=20,
    ).json()
    click.echo("test message sent." if confirm.get("ok") else f"send failed: {confirm}")


@main.command()
def doctor() -> None:
    """Check the environment. Run this first when something is wrong."""
    import shutil
    import sqlite3

    cfg, conn = _open()
    problems = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal problems
        if not ok:
            problems += 1
        click.echo(f"  [{'ok ' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")

    click.echo("binaries")
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        check(binary, path is not None, path or "not on PATH")

    click.echo("python packages")
    # yt-dlp is used as a library, so the version that matters is the importable
    # one, not whatever binary happens to be on PATH. A stale yt-dlp is the
    # single most likely cause of a silent pipeline stall, so surface its age.
    try:
        import yt_dlp

        version = getattr(yt_dlp, "version", None)
        stamp = getattr(version, "__version__", "unknown") if version else "unknown"
        age_days = None
        try:
            released = datetime.strptime(stamp[:10], "%Y.%m.%d").replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - released).days
        except ValueError:
            pass
        fresh = age_days is None or age_days < 45
        check(
            f"yt-dlp {stamp}",
            fresh,
            "" if fresh else f"{age_days} days old — run: uv pip install -U yt-dlp",
        )
    except ImportError:
        check("yt-dlp", False, "not installed")

    for module, extra in (
        ("structlog", ""), ("httpx", ""), ("yaml", ""), ("click", ""),
        ("opencc", ""), ("mlx", " (extra: asr)"), ("numpy", " (extra: asr)"),
        ("torch", " (extra: voice)"), ("speechbrain", " (extra: voice)"),
    ):
        try:
            __import__(module)
            check(module + extra, True)
        except ImportError:
            check(module + extra, False, "not installed")

    click.echo("speaker identification")
    if cfg.get("voice", "enabled", True):
        from .voice import DEFAULT_MARGIN, DEFAULT_THRESHOLD, voiceprints
        prints = voiceprints(conn)
        # No voiceprints is not an error -- the pipeline degrades to attributing
        # nobody, which is safe. But it silently means no track record can be
        # built, so say it plainly rather than reporting a clean bill of health.
        check(f"voiceprints enrolled: {len(prints)}", bool(prints),
              "" if prints else "none — every view will be unattributed")
        for name in sorted(prints):
            row = conn.execute("SELECT total_s FROM voiceprints WHERE speaker=?",
                               (name,)).fetchone()
            check(f"  {name}", True, f"{row['total_s'] / 60:.0f} min of voice")
        # A config that disagrees with the calibrated default would attribute at
        # an uncalibrated threshold, which is worse than not attributing at all.
        drift = (float(cfg.get("voice", "threshold", DEFAULT_THRESHOLD)) != DEFAULT_THRESHOLD
                 or float(cfg.get("voice", "margin", DEFAULT_MARGIN)) != DEFAULT_MARGIN)
        check("thresholds match calibration", not drift,
              "" if not drift else "config.toml differs from voice.py defaults")
        stale = conn.execute(
            "SELECT COUNT(*) FROM voiceprints WHERE model != ?",
            (__import__("ytdigest.voice", fromlist=["MODEL"]).MODEL,)).fetchone()[0]
        check("voiceprints match current model", stale == 0,
              "" if not stale else f"{stale} built with a different model — re-enrol")
    else:
        check("disabled in config.toml", True, "no speaker attribution")

    click.echo("storage")
    usage = shutil.disk_usage(cfg.data_dir)
    free_gb = usage.free / 2**30
    check(f"disk free {free_gb:.0f} GB", free_gb > 5, "" if free_gb > 5 else "under 5 GB")
    check("data dir writable", cfg.data_dir.exists())
    check("out dir writable", cfg.out_dir.exists())

    click.echo("database")
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    check("integrity", integrity == "ok", integrity)
    check("WAL mode", conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal")
    applied = [r["name"] for r in conn.execute("SELECT name FROM schema_migrations ORDER BY name")]
    check(f"migrations applied: {len(applied)}", bool(applied), ", ".join(applied))
    check("sqlite " + sqlite3.sqlite_version, True)

    click.echo("secrets")
    check("DEEPSEEK_API_KEY", cfg.secret("DEEPSEEK_API_KEY") is not None, "set in .env")
    if cfg.get("publish", "telegram", False):
        check("TELEGRAM_BOT_TOKEN", cfg.secret("TELEGRAM_BOT_TOKEN") is not None)
        check("TELEGRAM_CHAT_ID", cfg.secret("TELEGRAM_CHAT_ID") is not None)

    click.echo("scheduling")
    # Accept either the generic label or a namespaced one, since installs on a
    # shared machine usually follow that machine's own naming convention.
    home = Path.home() / "Library/LaunchAgents"
    label, plist = "com.ytdigest", home / "com.ytdigest.plist"
    for candidate in sorted(home.glob("*ytdigest*.plist")):
        label, plist = candidate.stem, candidate
        break
    check("LaunchAgent installed", plist.exists(), str(plist))
    if plist.exists():
        import subprocess

        loaded = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True, text=True,
        )
        check("LaunchAgent loaded", loaded.returncode == 0)

    click.echo("")
    if problems:
        click.echo(f"{problems} problem(s) found.")
        sys.exit(1)
    click.echo("all checks passed.")


if __name__ == "__main__":
    main()
