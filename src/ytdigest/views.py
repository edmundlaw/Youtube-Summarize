"""Extraction and storage of market views.

A "view" is one speaker's call on one instrument: direction, optionally a
level, optionally a horizon. It is the unit you backtest — not a video, not a
summary.

The rule that governs everything here: a level is stored as `verified` only if
it matches a number_ledger row for that video on value and unit. An unverified
level is still stored, because the direction and thesis remain useful, but it
is marked so a backtest can exclude it. Silently treating an unverified number
as tradeable is the failure this project exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import REPO_ROOT
from .db import now_iso, transaction
from .numbers import find_numbers

DIRECTIONS = {"long", "short", "neutral", "avoid", "exit"}

#: Generic role words the model writes when it will not commit to a name. They
#: are not attributions and must never end up in a track record as though they
#: were a person.
_GENERIC_SPEAKER = {"主持", "主持人", "嘉賓", "host", "guest", "主播"}
HORIZONS = {"intraday", "days", "weeks", "months", "quarters", "year"}
ENTRY_BASES = {"immediate", "on_rally", "on_dip", "on_break", "on_confirmation",
               "unspecified"}
STANCES = {"bullish", "bearish", "neutral"}

#: Rough horizon lengths, used to compute when a call can be judged.
HORIZON_DAYS = {
    "intraday": 1, "days": 5, "weeks": 21,
    "months": 63, "quarters": 126, "year": 252,
}


@dataclass
class View:
    speaker: str | None
    instrument_raw: str
    instrument: str | None
    asset_class: str | None
    direction: str
    conviction: str | None
    thesis: str
    reasoning: str | None
    level_type: str | None
    level_value: float | None
    level_unit: str | None
    ledger_id: int | None
    level_verified: bool
    horizon: str | None
    start_s: float
    entry_basis: str = "unspecified"
    condition: str | None = None
    stance: str | None = None


# --- speaker canonicalisation ----------------------------------------------


def _load_people(root: Path | None = None) -> tuple[dict[str, str], list[str]]:
    import yaml

    path = (root or REPO_ROOT) / "config" / "people.yaml"
    if not path.exists():
        return {}, []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    lookup: dict[str, str] = {}
    for canonical, spec in (data.get("people") or {}).items():
        display = (spec or {}).get("display") or canonical
        for alias in [canonical, display, *((spec or {}).get("aliases") or [])]:
            lookup[str(alias).strip().lower()] = display
    return lookup, [str(x).lower() for x in (data.get("not_people") or [])]


def canonical_speaker(name: str | None, root: Path | None = None) -> str | None:
    """Resolve a spoken/parsed name to one canonical form.

    The same person appears as 羅家聰, KC博士 and 羅家聰 KC 博士 depending on who
    is introducing them. Left alone, a track record splits across three names
    and counts none of them correctly. Anything unrecognised returns None —
    an unattributed view is useful, a misattributed one is worse than useless.
    """
    if not name:
        return None
    text = name.strip()
    lookup, not_people = _load_people(root)
    low = text.lower()
    if any(bad in low for bad in not_people) or "|" in text or len(text) > 30:
        return None
    if low in lookup:
        return lookup[low]
    # Longest containing alias wins: "羅家聰 KC 博士" must not resolve via "KC".
    best = None
    for alias, display in lookup.items():
        if alias and alias in low and (best is None or len(alias) > len(best[0])):
            best = (alias, display)
    return best[1] if best else None


# --- instrument mapping ----------------------------------------------------


def load_instruments(root: Path | None = None) -> dict:
    import yaml

    path = (root or REPO_ROOT) / "config" / "instruments.yaml"
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("instruments", {})


def sync_instruments(conn, root: Path | None = None) -> int:
    """Load the YAML mapping into the DB. Idempotent."""
    data = load_instruments(root)
    with transaction(conn):
        for symbol, spec in data.items():
            conn.execute(
                "INSERT INTO instruments (symbol, asset_class, display_name, currency, added_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET "
                "  asset_class=excluded.asset_class, display_name=excluded.display_name, "
                "  currency=excluded.currency",
                (symbol, spec.get("asset_class", "unknown"), spec.get("display_name"),
                 spec.get("currency"), now_iso()),
            )
            for alias in [symbol, *(spec.get("aliases") or [])]:
                conn.execute(
                    "INSERT INTO instrument_aliases (alias, symbol, added_at) VALUES (?,?,?) "
                    "ON CONFLICT(alias) DO UPDATE SET symbol=excluded.symbol",
                    (str(alias), symbol, now_iso()),
                )
    return len(data)


def resolve_instrument(conn, spoken: str) -> tuple[str | None, str | None]:
    """Map a spoken name to (symbol, asset_class). Longest alias wins.

    Longest-first matters: 恒生指數 must not be resolved by a shorter alias that
    happens to be a substring of it.
    """
    text = (spoken or "").strip()
    if not text:
        return None, None
    row = conn.execute(
        "SELECT i.symbol, i.asset_class FROM instrument_aliases a "
        "JOIN instruments i ON i.symbol = a.symbol WHERE a.alias = ?", (text,)
    ).fetchone()
    if row:
        return row["symbol"], row["asset_class"]
    best = None
    for alias in conn.execute(
        "SELECT a.alias, i.symbol, i.asset_class FROM instrument_aliases a "
        "JOIN instruments i ON i.symbol = a.symbol"
    ):
        if alias["alias"] in text and (best is None or len(alias["alias"]) > len(best["alias"])):
            best = alias
    return (best["symbol"], best["asset_class"]) if best else (None, None)


# --- level verification ----------------------------------------------------


#: A price level is almost never spoken with its unit — "跌到205" is how an
#: analyst says "down to USD 205". The ledger correctly records that as a
#: unit-less `count`, so demanding an exact unit match rejected nearly every
#: real level (2 of 19 on the first run). A bare count may therefore back a
#: money or index-points level; it may NOT back a percentage or a multiple,
#: where the unit carries the meaning.
_LEDGER_FOR = {
    "usd": {"usd", "count"}, "hkd": {"hkd", "count"}, "cny": {"cny", "count"},
    "points": {"count"}, "index": {"count"}, None: {"count"},
    "pct": {"pct"}, "bps": {"bps"}, "multiple": {"multiple"},
}


def verify_level(conn, video_id: str, value: float | None, unit: str | None):
    """Find the number_ledger row backing this level. Returns (id, verified).

    Verification is on the VALUE plus a compatible unit — the same rule the
    summary validator uses, kept deliberately parallel so the two cannot drift.
    A level that does not verify is still stored; it is simply excluded from a
    backtest by `level_verified = 0`.
    """
    if value is None:
        return None, False
    allowed = _LEDGER_FOR.get(unit, {unit, "count"} if unit else {"count"})
    rows = list(conn.execute(
        "SELECT id, unit, normalized FROM number_ledger WHERE video_id = ?", (video_id,)
    ))
    fallback = None
    for row in rows:
        try:
            stored = float(row["normalized"])
        except (TypeError, ValueError):
            continue
        if stored != value:
            continue
        if row["unit"] in allowed:
            return row["id"], True
        fallback = fallback or row["id"]
    # Value spoken, but under a unit that cannot back this kind of level.
    return fallback, False


def _first_number(text: str) -> tuple[float | None, str | None]:
    hits = [h for h in find_numbers(text or "") if h.value is not None]
    return (hits[0].value, hits[0].unit) if hits else (None, None)


# --- parsing the model's output --------------------------------------------


def parse_views(payload: dict, hosts: list[str] | None = None) -> list[View]:
    """Turn the summariser's `views` array into View objects.

    Anything malformed is dropped rather than guessed at — a half-parsed view
    is worse than a missing one, because it looks like data.
    """
    out: list[View] = []
    roster = {h.strip() for h in (hosts or []) if h.strip()}
    for item in payload.get("views") or []:
        if not isinstance(item, dict):
            continue
        direction = str(item.get("direction", "")).strip().lower()
        thesis = str(item.get("thesis", "")).strip()
        raw = str(item.get("instrument_raw") or item.get("instrument") or "").strip()
        if direction not in DIRECTIONS or not thesis or not raw:
            continue

        speaker = str(item.get("speaker") or "").strip() or None
        speaker = canonical_speaker(speaker) or (
            speaker if speaker in _GENERIC_SPEAKER else None)
        if speaker in _GENERIC_SPEAKER:
            # "主持" is a role, not a name. When the episode has exactly one
            # person on the roster it can only mean them; with two or more,
            # guessing would put words in someone's mouth, so drop it.
            speaker = (canonical_speaker(next(iter(roster)))
                       if len(roster) == 1 else None)
        # Attribution must come from the video's own host roster. A station
        # trailer for another programme names that show's host mid-episode, and
        # the model has been observed adopting the name.
        elif speaker and roster:
            allowed = {canonical_speaker(h) for h in roster} - {None}
            if allowed and speaker not in allowed:
                speaker = None

        value = item.get("level_value")
        try:
            value = float(value) if value is not None and value != "" else None
        except (TypeError, ValueError):
            value = None
        unit = (str(item.get("level_unit")).strip().lower()
                if item.get("level_unit") else None)
        if value is None and item.get("level"):
            value, unit = _first_number(str(item["level"]))

        horizon = str(item.get("horizon") or "").strip().lower() or None
        if horizon not in HORIZONS:
            horizon = None

        out.append(View(
            speaker=speaker,
            instrument_raw=raw,
            instrument=None,
            asset_class=None,
            direction=direction,
            conviction=(str(item.get("conviction")).strip().lower()
                        if item.get("conviction") else None),
            thesis=thesis,
            reasoning=str(item.get("reasoning") or "").strip() or None,
            level_type=(str(item.get("level_type")).strip().lower()
                        if item.get("level_type") else None),
            level_value=value,
            level_unit=unit,
            ledger_id=None,
            level_verified=False,
            horizon=horizon,
            start_s=_offset(item.get("ts")),
            entry_basis=_basis(item),
            condition=str(item.get("condition") or "").strip() or None,
            stance=(str(item.get("stance")).strip().lower()
                    if str(item.get("stance") or "").strip().lower() in STANCES
                    else _implied_stance(direction)),
        ))
    return out


#: Trigger words that make a call conditional. Recording such a call as an
#: immediate one would backtest an entry the speaker explicitly warned against.
_BASIS_HINTS = [
    ("on_rally", ["反彈", "彈起", "彈上", "回升", "if it bounces", "彈嘅話", "彈的話"]),
    ("on_dip", ["回落", "跌到", "調整到", "回調", "落到"]),
    ("on_break", ["跌穿", "升穿", "突破", "破位", "穿咗"]),
    ("on_confirmation", ["確認", "見到訊號", "有信號", "企穩"]),
]


def _basis(item: dict) -> str:
    """Decide whether the call is immediate or waits for a trigger."""
    stated = str(item.get("entry_basis") or "").strip().lower()
    if stated in ENTRY_BASES:
        return stated
    haystack = " ".join(str(item.get(k) or "") for k in
                        ("condition", "thesis", "reasoning"))
    for basis, hints in _BASIS_HINTS:
        if any(h in haystack for h in hints):
            return basis
    return "unspecified"


def _implied_stance(direction: str) -> str | None:
    return {"long": "bullish", "short": "bearish", "avoid": "bearish",
            "exit": "bearish", "neutral": "neutral"}.get(direction)


def _offset(ts) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = re.findall(r"\d+", str(ts))
    if not parts:
        return 0.0
    nums = [int(p) for p in parts]
    if len(nums) >= 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return float(nums[0])


# --- storage ---------------------------------------------------------------


def _voice_speaker(conn, video_id: str, start_s: float) -> str | None:
    """Enrolled speaker covering this timestamp, per stored identification."""
    row = conn.execute(
        "SELECT speaker FROM segment_speakers WHERE video_id = ? AND start_s <= ? "
        "AND end_s >= ? AND speaker IS NOT NULL ORDER BY start_s DESC LIMIT 1",
        (video_id, start_s, start_s),
    ).fetchone()
    return row["speaker"] if row else None


def store_views(conn, video: dict, views: list[View], summary_id: int | None,
                prompt_version: str) -> int:
    """Resolve, verify and persist. Returns the number of rows written."""
    published = video.get("published_at") or now_iso()
    try:
        base = datetime.fromisoformat(published)
    except ValueError:
        base = datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)

    # Voice identification, where it has run, outranks whatever name the model
    # put on a view. One is a measurement against an enrolled voiceprint; the
    # other is inference from unlabelled text. When they disagree the
    # measurement wins, and when the voice could not be identified the view is
    # left unattributed rather than falling back to the model's guess -- that
    # fallback is exactly how a wrong name reaches the track record.
    identified = conn.execute(
        "SELECT COUNT(*) FROM segment_speakers WHERE video_id = ?",
        (video["id"],),
    ).fetchone()[0] > 0

    written = 0
    with transaction(conn):
        for view in views:
            if identified:
                voiced = _voice_speaker(conn, video["id"], view.start_s)
                speaker = voiced
                attribution = "voice" if voiced else "none"
            else:
                speaker = view.speaker
                attribution = "guessed" if view.speaker else "none"

            symbol, asset_class = resolve_instrument(conn, view.instrument_raw)
            ledger_id, verified = verify_level(
                conn, video["id"], view.level_value, view.level_unit
            )
            stated_at = (base + timedelta(seconds=view.start_s)).astimezone(UTC)
            ends_at = None
            if view.horizon:
                ends_at = (stated_at + timedelta(
                    days=HORIZON_DAYS.get(view.horizon, 21) * 1.4
                )).isoformat(timespec="seconds")

            conn.execute(
                "INSERT INTO views (video_id, channel_id, speaker, stated_at, start_s, "
                " instrument, instrument_raw, asset_class, direction, conviction, thesis, "
                " reasoning, level_type, level_value, level_unit, ledger_id, level_verified, "
                " horizon, horizon_ends_at, outcome, summary_id, prompt_version, created_at, "
                " entry_basis, condition, stance, attribution) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?,?,?) "
                "ON CONFLICT DO UPDATE SET "
                "  thesis=excluded.thesis, reasoning=excluded.reasoning, "
                "  instrument=excluded.instrument, level_verified=excluded.level_verified, "
                "  ledger_id=excluded.ledger_id, summary_id=excluded.summary_id, "
                "  attribution=excluded.attribution",
                (video["id"], video["channel_id"], speaker,
                 stated_at.isoformat(timespec="seconds"), view.start_s,
                 symbol, view.instrument_raw, asset_class, view.direction,
                 view.conviction, view.thesis, view.reasoning, view.level_type,
                 view.level_value, view.level_unit, ledger_id, int(verified),
                 view.horizon, ends_at, summary_id, prompt_version, now_iso(),
                 view.entry_basis, view.condition, view.stance, attribution),
            )
            written += 1
    return written
