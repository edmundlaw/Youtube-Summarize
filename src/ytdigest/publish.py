"""Markdown output and Telegram delivery.

Frontmatter carries full lineage: given a summary you can name the transcript,
the ASR model, the prompt version and the LLM that produced it.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import httpx

from .config import Config
from .db import RETRYABLE, StageError
from .validator import Check

TG_LIMIT = 4096


def _mmss(value: str | float) -> str:
    if isinstance(value, str):
        return value[-5:] if len(value) > 5 else value
    return f"{int(value) // 60:02d}:{int(value) % 60:02d}"


def _mark(figure: str, checks: list[Check]) -> str:
    """Figures that did not cleanly verify are always rendered with a warning."""
    for check in checks:
        if check.figure == figure and check.verdict != "ok":
            return f"⚠︎{figure}"
    return figure


def render_markdown(
    *,
    video: dict,
    payload: dict,
    checks: list[Check],
    state: str,
    meta: dict,
) -> str:
    unverified = [c for c in checks if c.verdict == "missing"]
    flagged = [c for c in checks if c.verdict == "flagged"]

    front = {
        "video_id": video["id"],
        "title": video["title"],
        "channel": video.get("channel_title", ""),
        "published": video.get("published_at", "")[:10],
        "duration": video.get("duration_s"),
        "lang": meta.get("lang"),
        "url": f"https://youtu.be/{video['id']}",
        "transcript_source": meta.get("source"),
        "asr_model": meta.get("model_id"),
        "prompt_version": meta.get("prompt_version"),
        "summary_model": meta.get("summary_model"),
        "transcript_sha": meta.get("transcript_sha"),
        "validator": state,
        "tags": ["ytdigest", "finance"],
    }
    lines = ["---"]
    for key, value in front.items():
        if value in (None, ""):
            continue
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(value)}]")
        elif isinstance(value, str) and (":" in value or "#" in value):
            lines.append(f'{key}: "{value}"')
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")

    if unverified:
        lines += [
            "> [!warning] 未核實數字",
            "> 以下數字喺字幕原文核對唔到，已用 ⚠︎ 標示，請自行核實：",
            "> " + "、".join(dict.fromkeys(c.figure for c in unverified)),
            "",
        ]

    lines.append(f"# {video['title']}")
    lines.append("")

    if payload.get("actionable"):
        lines += ["## 可操作", ""]
        for item in payload["actionable"]:
            ticker = f"**{item['ticker']}** " if item.get("ticker") else ""
            claim = _annotate(item.get("claim", ""), checks)
            lines.append(f"- `[{_mmss(item.get('ts', ''))}]` {ticker}{claim}")
        lines.append("")

    if payload.get("theses"):
        lines += ["## 論點", ""]
        for item in payload["theses"]:
            lines.append(
                f"- `[{_mmss(item.get('ts', ''))}]` "
                f"**{_annotate(item.get('thesis', ''), checks)}**"
            )
            if item.get("reasoning"):
                lines.append(f"  - {_annotate(item['reasoning'], checks)}")
        lines.append("")

    if payload.get("disagreements"):
        lines += ["## 分歧", ""]
        for item in payload["disagreements"]:
            lines.append(
                f"- `[{_mmss(item.get('ts', ''))}]` "
                f"{_annotate(item.get('detail', ''), checks)}"
            )
        lines.append("")

    if payload.get("risks"):
        lines += ["## 風險", ""]
        for item in payload["risks"]:
            lines.append(
                f"- `[{_mmss(item.get('ts', ''))}]` "
                f"{_annotate(item.get('risk', ''), checks)}"
            )
        lines.append("")

    if payload.get("numbers"):
        lines += ["## 數字", "", "| 數字 | 出處 | 時間 | 核實 |", "|---|---|---|---|"]
        by_figure = {c.figure: c for c in checks}
        for item in payload["numbers"]:
            figure = str(item.get("figure", ""))
            check = by_figure.get(figure)
            status = {"ok": "✓", "flagged": "⚠︎ 需核實", "missing": "✗ 核對唔到"}.get(
                check.verdict if check else "", "—"
            )
            lines.append(
                f"| {_mark(figure, checks)} | {item.get('context', '')} "
                f"| `{_mmss(item.get('ts', ''))}` | {status} |"
            )
        lines.append("")

    if flagged or unverified:
        lines += ["## 核實備註", ""]
        for check in flagged + unverified:
            lines.append(f"- `{check.figure}` — {check.reason}")
        lines.append("")

    return "\n".join(lines)


def _annotate(text: str, checks: list[Check]) -> str:
    """Mark each unverified figure once.

    Iterating checks directly marked a figure once per matching check, so a
    number appearing in six places came out as ⚠︎⚠︎⚠︎⚠︎⚠︎⚠︎2022年. Deduplicate
    first, mark longest-first so a short figure that is a substring of a longer
    one does not corrupt it, and never double-prefix.
    """
    figures = sorted(
        {c.figure for c in checks if c.verdict != "ok" and c.figure},
        key=len, reverse=True,
    )
    for figure in figures:
        text = re.sub(f"(?<!\u26a0\ufe0e){re.escape(figure)}", f"\u26a0\ufe0e{figure}", text)
    return text


def write_markdown(out_dir: Path, video: dict, content: str) -> Path:
    """Filenames are always <date>--<video_id>. Titles contain CJK, emoji,
    slashes and 200 characters of clickbait; they never go in a path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    date = (video.get("published_at") or "")[:10] or "undated"
    path = out_dir / f"{date}--{video['id']}.md"
    path.write_text(content, encoding="utf-8")
    return path


# --- Telegram --------------------------------------------------------------


def _esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def render_telegram(video: dict, payload: dict, checks: list[Check], state: str) -> str:
    unverified = [c for c in checks if c.verdict == "missing"]
    lines = [
        f"🎬 <b>{_esc(video['title'])[:120]}</b>",
        f"{_esc(video.get('channel_title', ''))} · "
        f"{_esc((video.get('published_at') or '')[:10])}",
        f"<a href='https://youtu.be/{video['id']}'>影片</a>",
    ]
    if unverified:
        lines += ["", "⚠️ <b>有數字核對唔到</b>：" +
                  _esc("、".join(c.figure for c in unverified))]

    for header, key, fmt in (
        ("📌 可操作", "actionable", lambda i: (
            (f"<b>{_esc(i['ticker'])}</b> " if i.get("ticker") else "")
            + _esc(i.get("claim", "")))),
        ("🧠 論點", "theses", lambda i: (
            f"<b>{_esc(i.get('thesis', ''))}</b>"
            + (f"\n　└ {_esc(i.get('reasoning', ''))}" if i.get("reasoning") else ""))),
        ("⚔️ 分歧", "disagreements", lambda i: _esc(i.get("detail", ""))),
        ("⚠️ 風險", "risks", lambda i: _esc(i.get("risk", ""))),
    ):
        items = payload.get(key) or []
        if not items:
            continue
        lines += ["", f"<b>{header}</b>"]
        for item in items[:6]:
            lines.append(f"<code>[{_mmss(item.get('ts', ''))}]</code> {fmt(item)}")

    if payload.get("numbers"):
        lines += ["", "<b>🔢 數字</b>　✓核對到　⚠︎需核實"]
        by_figure = {c.figure: c for c in checks}
        for item in payload["numbers"][:12]:
            figure = str(item.get("figure", ""))
            check = by_figure.get(figure)
            tick = "✓" if check and check.verdict == "ok" else "⚠︎"
            lines.append(f"{tick} <b>{_esc(figure)}</b> {_esc(item.get('context', ''))}")

    lines += ["", f"<i>validator: {state}</i>"]
    text = "\n".join(lines)
    if len(text) > TG_LIMIT:
        text = text[: TG_LIMIT - 40].rsplit("\n", 1)[0] + "\n<i>…(截斷)</i>"
    return text


def send_telegram(cfg: Config, text: str) -> None:
    token = cfg.secret("TELEGRAM_BOT_TOKEN")
    chat_id = cfg.secret("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise StageError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", "permanent")
    body: dict = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    thread = cfg.secret("TELEGRAM_MESSAGE_THREAD_ID")
    if thread:
        body["message_thread_id"] = int(thread)
    try:
        response = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage", json=body, timeout=30
        )
        payload = response.json()
    except Exception as exc:
        raise StageError(f"telegram request failed: {exc}", RETRYABLE) from exc
    if not payload.get("ok"):
        raise StageError(f"telegram rejected: {payload.get('description')}", RETRYABLE)


def notify(cfg: Config, text: str) -> None:
    """Best-effort operational notice. Never fails a stage."""
    try:
        send_telegram(cfg, text)
    except Exception:
        pass


def dump_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
