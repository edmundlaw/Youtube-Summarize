"""The resolver decides whether a person was right. Every one of these tests
exists because getting it wrong would misrepresent someone's ability.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from ytdigest import db as D
from ytdigest.resolver import (
    HIT, MISSED, PENDING, UNRESOLVABLE, VOID, resolve_view, scorecard,
)


@pytest.fixture
def conn():
    path = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    c = D.open_db(path, pathlib.Path("migrations"))
    c.execute("INSERT INTO channels (id,title,added_at) VALUES ('UC1','c',?)",
              (D.now_iso(),))
    c.execute("INSERT INTO videos (id,channel_id,title,published_at,discovered_at,"
              "status) VALUES ('v1','UC1','t','2026-01-01','2026-01-01','done')")
    c.execute("INSERT INTO instruments (symbol,asset_class,added_at) "
              "VALUES ('^HSI','index',?)", (D.now_iso(),))
    return c


def add_bars(conn, symbol, bars):
    for date, low, high in bars:
        conn.execute(
            "INSERT INTO prices (symbol,date,open,high,low,close,source,fetched_at) "
            "VALUES (?,?,?,?,?,?,'test',?)",
            (symbol, date, (high + low) / 2, high, low, (high + low) / 2, D.now_iso()))


def add_view(conn, **kw):
    fields = {
        "video_id": "v1", "channel_id": "UC1", "speaker": "someone",
        "stated_at": "2026-01-01T00:00:00+00:00", "start_s": 0.0,
        "instrument": "^HSI", "instrument_raw": "恒指", "direction": "long",
        "thesis": "t", "level_value": 26000.0, "level_unit": "points",
        "level_verified": 1, "horizon": "weeks", "entry_basis": "immediate",
        "outcome": "pending", "created_at": D.now_iso(),
    }
    fields.update(kw)
    cols = ",".join(fields)
    conn.execute(f"INSERT INTO views ({cols}) VALUES ({','.join('?' * len(fields))})",
                 tuple(fields.values()))
    return conn.execute("SELECT * FROM views ORDER BY id DESC LIMIT 1").fetchone()


# --- refusals: these must never be scored against a speaker ----------------

def test_unmapped_instrument_is_unresolvable(conn):
    v = add_view(conn, instrument=None)
    assert resolve_view(conn, v).outcome == UNRESOLVABLE


def test_unverified_level_is_unresolvable(conn):
    """If the number could not be traced to the transcript, the call is not
    trustworthy input — grading it would launder a bad figure into a score."""
    v = add_view(conn, level_verified=0)
    assert resolve_view(conn, v).outcome == UNRESOLVABLE


def test_missing_horizon_is_unresolvable(conn):
    """Choosing a horizon for the speaker decides whether they were right."""
    v = add_view(conn, horizon=None)
    verdict = resolve_view(conn, v)
    assert verdict.outcome == UNRESOLVABLE
    assert "horizon" in verdict.note


def test_direction_only_view_is_unresolvable(conn):
    v = add_view(conn, level_value=None)
    assert resolve_view(conn, v).outcome == UNRESOLVABLE


def test_neutral_call_is_not_a_trade(conn):
    v = add_view(conn, direction="neutral")
    assert resolve_view(conn, v).outcome == UNRESOLVABLE


def test_no_price_data_is_pending_not_a_verdict(conn):
    """Absence of data must never become 'missed'."""
    v = add_view(conn)
    assert resolve_view(conn, v).outcome == PENDING


def test_open_horizon_is_pending(conn):
    v = add_view(conn, stated_at="2099-01-01T00:00:00+00:00")
    assert resolve_view(conn, v).outcome == PENDING


# --- actual grading --------------------------------------------------------

def test_long_target_reached_is_hit(conn):
    add_bars(conn, "^HSI", [("2026-01-05", 25000, 25500),
                            ("2026-01-12", 25800, 26100)])
    v = add_view(conn, direction="long", level_value=26000.0)
    assert resolve_view(conn, v).outcome == HIT


def test_long_target_not_reached_is_missed(conn):
    add_bars(conn, "^HSI", [("2026-01-05", 25000, 25500),
                            ("2026-01-12", 25200, 25600)])
    v = add_view(conn, direction="long", level_value=26000.0)
    assert resolve_view(conn, v).outcome == MISSED


def test_short_target_uses_the_low(conn):
    add_bars(conn, "^HSI", [("2026-01-05", 24000, 25500),
                            ("2026-01-12", 23500, 24800)])
    v = add_view(conn, direction="short", level_value=23800.0)
    assert resolve_view(conn, v).outcome == HIT


# --- the conditional case that prompted all of this ------------------------

def test_conditional_short_that_never_rallied_is_void_not_missed(conn):
    """KC on SK Hynix: sell IF it bounces. If it only ever fell, he never told
    anyone to act — recording that as a failed short measures a trade he did
    not recommend."""
    add_bars(conn, "^HSI", [("2026-01-05", 24000, 24500),
                            ("2026-01-12", 23000, 23800)])   # no rally
    v = add_view(conn, direction="short", entry_basis="on_rally",
                 level_value=22000.0)
    verdict = resolve_view(conn, v)
    assert verdict.outcome == VOID
    assert "never fired" in verdict.note


def test_conditional_short_graded_only_after_the_rally(conn):
    add_bars(conn, "^HSI", [("2026-01-05", 24000, 24500),
                            ("2026-01-08", 24800, 25500),    # rally: trigger
                            ("2026-01-15", 21500, 24000)])   # then falls
    v = add_view(conn, direction="short", entry_basis="on_rally",
                 level_value=22000.0)
    assert resolve_view(conn, v).outcome == HIT


# --- scorecard honesty -----------------------------------------------------

def test_thin_record_reports_no_hit_rate(conn):
    """Two graded calls and forty unresolvable ones is not a 100% record.
    Showing a number there would be worse than showing nothing."""
    add_bars(conn, "^HSI", [("2026-01-05", 25000, 26500)])
    add_view(conn, speaker="thin", outcome="hit")
    add_view(conn, speaker="thin", outcome="unresolvable", level_value=25999.0)
    rows = {r["speaker"]: r for r in scorecard(conn, min_graded=5)}
    assert rows["thin"]["hit_rate"] is None
    assert rows["thin"]["unresolvable"] == 1


def test_void_and_unresolvable_excluded_from_hit_rate(conn):
    for i in range(6):
        add_view(conn, speaker="fair", outcome="hit", level_value=100.0 + i)
    for i in range(4):
        add_view(conn, speaker="fair", outcome="void", level_value=200.0 + i)
    for i in range(3):
        add_view(conn, speaker="fair", outcome="unresolvable", level_value=300.0 + i)
    row = {r["speaker"]: r for r in scorecard(conn)}["fair"]
    assert row["graded"] == 6            # voids and unresolvables not counted
    assert row["hit_rate"] == 1.0
    assert row["void"] == 4 and row["unresolvable"] == 3
