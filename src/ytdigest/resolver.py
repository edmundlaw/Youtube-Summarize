"""Grading market views against actual prices.

The point of this module is to be *hard to mislead with*. A track record is
only useful if the things it counts are things the speaker actually said, and
if the things it cannot judge are excluded loudly rather than guessed at.

So the governing rule is: **refuse rather than guess.** Every view is assigned
exactly one outcome, and three of the five are refusals:

    hit           the stated level was reached inside the horizon
    missed        the horizon elapsed and it was not
    void          a conditional call whose trigger never fired — the speaker
                  never told you to act, so there is nothing to grade
    unresolvable  something needed to judge it is missing or ambiguous
    pending       the horizon has not elapsed yet, or prices are not loaded

`unresolvable` and `void` are not failures of the speaker and must never be
counted against them. Every row carries `outcome_note` saying exactly why it
was graded the way it was, so no verdict is a black box.

Deliberately NOT done here:
* No inference of a horizon that was not stated. Picking one for the speaker
  decides whether they were right, which is the whole question.
* No grading of unverified levels. If the number could not be traced to the
  transcript, the call itself is not trustworthy input.
* No partial credit, no "close enough" band. A target is reached or it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .db import now_iso, transaction
from .views import HORIZON_DAYS

HIT = "hit"
MISSED = "missed"
VOID = "void"
UNRESOLVABLE = "unresolvable"
PENDING = "pending"

#: Outcomes that count toward a hit rate. The others are excluded from the
#: denominator, which is why they must be reported alongside any score.
GRADED = {HIT, MISSED}


@dataclass
class Verdict:
    outcome: str
    value: float | None
    note: str


def _bars(conn, symbol: str, start: str, end: str) -> list:
    return list(conn.execute(
        "SELECT date, open, high, low, close FROM prices "
        "WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date",
        (symbol, start, end),
    ))


def _window(view) -> tuple[str, str] | None:
    """The date range over which the call should be judged."""
    if not view["horizon"] or view["horizon"] not in HORIZON_DAYS:
        return None
    start = datetime.fromisoformat(view["stated_at"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    # Calendar days, generously, since HORIZON_DAYS counts trading days.
    end = start + timedelta(days=HORIZON_DAYS[view["horizon"]] * 1.5)
    return start.date().isoformat(), end.date().isoformat()


def _trigger_index(bars: list, basis: str, reference: float,
                   level: float | None) -> int | None:
    """First bar at which a conditional call becomes live.

    A conditional call is not advice to act now. Grading it from the moment it
    was spoken measures an entry the speaker explicitly did not recommend —
    which is how a correct 'sell into any bounce' gets recorded as a bad short
    when the stock simply kept falling.
    """
    # Start at the bar AFTER the statement. Comparing the statement day's own
    # high against its own close makes every call self-trigger on day one,
    # which silently turns "sell if it bounces" back into "sell now" — the
    # exact error this function exists to prevent.
    for index, bar in enumerate(bars[1:], start=1):
        if basis == "on_rally":
            # A close above the reference, not an intraday poke: a wick through
            # is noise, and grading a real call on noise is how a track record
            # becomes meaningless.
            if bar["close"] is not None and bar["close"] > reference:
                return index
        elif basis == "on_dip":
            if bar["close"] is not None and bar["close"] < reference:
                return index
        elif basis == "on_break" and level is not None:
            # A break of a level IS an intraday event, so high/low is right here.
            if bar["high"] is not None and bar["high"] >= level:
                return index
            if bar["low"] is not None and bar["low"] <= level:
                return index
    return None


def resolve_view(conn, view) -> Verdict:
    """Grade one view. Returns a Verdict; never raises on bad data."""
    if not view["instrument"]:
        return Verdict(UNRESOLVABLE, None, "instrument not mapped to a symbol")
    if view["level_value"] is None:
        return Verdict(UNRESOLVABLE, None,
                       "no price level stated — direction-only view")
    if not view["level_verified"]:
        return Verdict(UNRESOLVABLE, None,
                       "level not traceable to the transcript; not trusted as input")
    if view["direction"] not in {"long", "short"}:
        return Verdict(UNRESOLVABLE, None,
                       f"direction '{view['direction']}' is not a testable trade")

    window = _window(view)
    if window is None:
        return Verdict(UNRESOLVABLE, None,
                       "no horizon stated — choosing one would decide the answer")
    start, end = window

    if end > datetime.now(UTC).date().isoformat():
        return Verdict(PENDING, None, f"horizon runs to {end}")

    bars = _bars(conn, view["instrument"], start, end)
    if not bars:
        return Verdict(PENDING, None,
                       f"no price data for {view['instrument']} in {start}..{end}")

    level = float(view["level_value"])
    basis = view["entry_basis"] or "unspecified"
    reference = bars[0]["close"]

    if basis in {"on_rally", "on_dip", "on_break"}:
        index = _trigger_index(bars, basis, reference, level)
        if index is None:
            return Verdict(VOID, None,
                           f"conditional on {basis}; trigger never fired, so the "
                           "speaker never advised acting — not counted either way")
        bars = bars[index:]

    highs = [b["high"] for b in bars if b["high"] is not None]
    lows = [b["low"] for b in bars if b["low"] is not None]
    if not highs or not lows:
        return Verdict(PENDING, None, "price bars missing high/low")

    if view["direction"] == "long":
        reached, extreme = max(highs) >= level, max(highs)
    else:
        reached, extreme = min(lows) <= level, min(lows)

    side = "high" if view["direction"] == "long" else "low"
    if reached:
        return Verdict(HIT, extreme,
                       f"{side} {extreme:g} reached {level:g} by {bars[-1]['date']}")
    return Verdict(MISSED, extreme,
                   f"best {side} was {extreme:g}, never reached {level:g} by {end}")


def resolve_all(conn, force: bool = False) -> dict[str, int]:
    """Grade every view that is not already settled. Returns outcome counts."""
    where = "" if force else " WHERE outcome IN ('pending','') OR outcome IS NULL"
    rows = list(conn.execute(f"SELECT * FROM views{where}"))
    counts: dict[str, int] = {}
    with transaction(conn):
        for view in rows:
            verdict = resolve_view(conn, view)
            counts[verdict.outcome] = counts.get(verdict.outcome, 0) + 1
            conn.execute(
                "UPDATE views SET outcome=?, outcome_value=?, outcome_note=?, "
                "resolved_at=? WHERE id=?",
                (verdict.outcome, verdict.value, verdict.note,
                 now_iso() if verdict.outcome != PENDING else None, view["id"]),
            )
    return counts


def scorecard(conn, min_graded: int = 5) -> list[dict]:
    """Per-speaker record, reporting what was excluded and why.

    A hit rate quoted without its exclusions is misleading: a speaker with two
    graded calls and forty unresolvable ones has no measurable record, and
    presenting '100%' for them would be worse than presenting nothing. Anyone
    below `min_graded` is returned with `verdict=None` and must be displayed as
    'not enough data', never as a number.
    """
    out: list[dict] = []
    for row in conn.execute(
        "SELECT COALESCE(speaker,'(unattributed)') AS speaker, "
        "  SUM(outcome='hit') AS hit, SUM(outcome='missed') AS missed, "
        "  SUM(outcome='void') AS void, SUM(outcome='unresolvable') AS unresolvable, "
        "  SUM(outcome='pending') AS pending, COUNT(*) AS total "
        "FROM views GROUP BY speaker ORDER BY total DESC"
    ):
        graded = (row["hit"] or 0) + (row["missed"] or 0)
        out.append({
            "speaker": row["speaker"],
            "hit": row["hit"] or 0,
            "missed": row["missed"] or 0,
            "void": row["void"] or 0,
            "unresolvable": row["unresolvable"] or 0,
            "pending": row["pending"] or 0,
            "total": row["total"],
            "graded": graded,
            # None means "no measurable record", NOT zero.
            "hit_rate": (row["hit"] / graded) if graded >= min_graded else None,
        })
    return out
