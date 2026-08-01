"""YouTube discovery and fetching via yt-dlp.

yt-dlp is used as a library, not a subprocess: the importable version is the one
that matters, and a stale copy is the single most likely cause of a silent
pipeline stall.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from ..db import PERMANENT, RETRYABLE, StageError
from ..interfaces import FetchResult, VideoRef
from ..vtt import is_usable_subtitle_track

RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}

# Errors that must never be retried. Retrying a members-only video burns hours;
# retrying a bot check actively worsens the block.
_PERMANENT_SIGNS = (
    "private video", "video unavailable", "members-only", "members only",
    "removed by the uploader", "age-restricted", "sign in to confirm",
    "this video is not available", "copyright",
)


def _classify(exc: Exception) -> str:
    text = str(exc).lower()
    return PERMANENT if any(s in text for s in _PERMANENT_SIGNS) else RETRYABLE


def _ydl(**overrides):
    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "extract_flat": False,
    }
    options.update(overrides)
    return YoutubeDL(options)


def resolve_channel(url: str) -> tuple[str, str]:
    """Resolve a channel URL, @handle or bare id to (channel_id, title)."""
    if re.fullmatch(r"UC[\w-]{22}", url):
        url = f"https://www.youtube.com/channel/{url}"
    elif not url.startswith("http"):
        url = f"https://www.youtube.com/@{url.lstrip('@')}"

    with _ydl(extract_flat="in_playlist", playlist_items="0") as ydl:
        info = ydl.extract_info(url, download=False)
    channel_id = info.get("channel_id") or info.get("uploader_id") or info.get("id")
    title = info.get("channel") or info.get("uploader") or info.get("title") or channel_id
    if not channel_id or not channel_id.startswith("UC"):
        raise ValueError(f"could not determine channel id (got {channel_id!r})")
    return channel_id, title


def discover(channel_id: str, backfill_days: int = 7) -> list[VideoRef]:
    """List recent uploads from the channel RSS feed.

    RSS is used rather than a playlist scrape: it is one cheap request, it is
    not rate limited in practice, and it does not look like scraping.
    """
    try:
        response = httpx.get(RSS.format(channel_id), timeout=30,
                             follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        raise StageError(f"RSS fetch failed: {exc}", RETRYABLE) from exc

    cutoff = datetime.now(UTC) - timedelta(days=backfill_days)
    refs: list[VideoRef] = []
    for entry in ET.fromstring(response.text).findall("atom:entry", _NS):
        vid = entry.findtext("yt:videoId", namespaces=_NS)
        published = entry.findtext("atom:published", namespaces=_NS)
        title = entry.findtext("atom:title", namespaces=_NS) or ""
        if not vid or not published:
            continue
        if datetime.fromisoformat(published) < cutoff:
            continue
        refs.append(
            VideoRef(id=vid, channel_id=channel_id, title=title, published_at=published)
        )
    return refs


def list_uploads(channel_id: str, limit: int = 500,
                 since: str | None = None) -> list[VideoRef]:
    """Every upload on a channel, newest first — not just the RSS window.

    RSS carries roughly the last fifteen uploads, which is about a fortnight for
    a daily show. That is enough to keep up and nowhere near enough to build a
    record of what someone has said, so history has to come from the uploads
    playlist instead.

    Listed flat: yt-dlp returns one entry per video without opening any of them,
    so eighty uploads cost about a second and no downloads.

    **A flat listing carries no date and no duration** — only id and title, with
    every other field null. `published_at` therefore comes back None, and the
    caller must probe the videos it actually wants. Defaulting the date to "now"
    was tried and is actively dangerous: `stated_at` is derived from it, so
    every backfilled opinion would be dated to the day it was imported and
    graded against the wrong price window entirely.

    `since` is accepted but can only be applied by the caller, after probing.
    """
    options = {
        "extract_flat": "in_playlist",
        "playlistend": limit,
        "skip_download": True,
    }
    url = f"https://www.youtube.com/channel/{channel_id}/videos"
    try:
        with _ydl(**options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise StageError(f"upload listing failed: {exc}", _classify(exc)) from exc

    refs: list[VideoRef] = []
    for entry in (info or {}).get("entries") or []:
        vid = entry.get("id")
        if not vid:
            continue
        refs.append(VideoRef(
            id=vid, channel_id=channel_id, title=entry.get("title") or "",
            published_at=None,
        ))
    return refs


def probe(video_id: str) -> dict:
    """Metadata for one video, without downloading anything."""
    try:
        with _ydl() as ydl:
            return ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
    except Exception as exc:
        raise StageError(f"probe failed: {exc}", _classify(exc)) from exc


def pick_subtitle_track(info: dict, lang_hint: str | None) -> tuple[str, str] | None:
    """Choose the best subtitle track. Returns (source_kind, lang_code).

    Human-uploaded subtitles beat machine ones, but `live_chat` is offered
    alongside them and is a chat replay, not a transcript — including it yields
    a plausible-looking 'transcript' of viewer comments.
    """
    manual = {k: v for k, v in (info.get("subtitles") or {}).items()
              if is_usable_subtitle_track(k)}
    auto = {k: v for k, v in (info.get("automatic_captions") or {}).items()
            if is_usable_subtitle_track(k)}

    preferred = [lang_hint] if lang_hint else []
    preferred += ["yue-orig", "yue", "zh-Hant", "zh-HK", "zh", "en"]

    for code in preferred:
        if code and code in manual:
            return "manual_subs", code
    for code in preferred:
        if code and code in auto:
            return "auto_subs", code
    if manual:
        return "manual_subs", next(iter(manual))
    if auto:
        return "auto_subs", next(iter(auto))
    return None


def fetch_subtitles(video_id: str, lang: str, kind: str, dest_dir: Path) -> Path:
    """Download one subtitle track as VTT. Returns the written path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = dest_dir / video_id
    options = {
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": str(stem) + ".%(ext)s",
        "writesubtitles": kind == "manual_subs",
        "writeautomaticsub": kind == "auto_subs",
    }
    try:
        with _ydl(**options) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as exc:
        raise StageError(f"subtitle download failed: {exc}", _classify(exc)) from exc

    for candidate in (
        dest_dir / f"{video_id}.{lang}.vtt",
        *sorted(dest_dir.glob(f"{video_id}.*.vtt")),
    ):
        if candidate.exists():
            return candidate
    raise StageError("yt-dlp reported success but wrote no .vtt", RETRYABLE)


def fetch_audio(video_id: str, dest_dir: Path) -> Path:
    """Download audio as 16 kHz mono wav, ready for ASR."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{video_id}.wav"
    options = {
        "skip_download": False,
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
        "postprocessor_args": {"ffmpeg": ["-ac", "1", "-ar", "16000"]},
    }
    try:
        with _ydl(**options) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as exc:
        raise StageError(f"audio download failed: {exc}", _classify(exc)) from exc
    if not out.exists():
        raise StageError("audio extraction produced no wav", RETRYABLE)
    return out


def fetch(video_id: str, dest_dir: Path, lang_hint: str | None = None) -> FetchResult:
    """Prefer subtitles; fall back to audio for ASR when none exist."""
    info = probe(video_id)
    track = pick_subtitle_track(info, lang_hint)
    if track is not None:
        kind, lang = track
        path = fetch_subtitles(video_id, lang, kind, dest_dir)
        return FetchResult(kind=kind, path=path, duration_s=info.get("duration"))
    return FetchResult(
        kind="audio", path=fetch_audio(video_id, dest_dir),
        duration_s=info.get("duration"),
    )
