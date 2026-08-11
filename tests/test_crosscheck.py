import pathlib
import tempfile

import pytest

from ytdigest import db as D


@pytest.fixture
def conn():
    path = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    return D.open_db(path, pathlib.Path("migrations"))


def test_ledger_carries_the_second_reading(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(number_ledger)")}
    assert {"asr_normalized", "crosscheck", "asr_model"} <= cols


def test_existing_rows_default_to_never_checked(conn):
    """A null crosscheck must mean 'nothing has looked at this', not 'agreed'.
    Every row already in production is in this state."""
    conn.execute(
        "INSERT INTO channels (id, title, enabled, added_at) "
        "VALUES ('c', 'Channel', 1, '2026-01-01')")
    conn.execute(
        "INSERT INTO videos (id, channel_id, title, published_at, discovered_at, status) "
        "VALUES ('v', 'c', 't', '2026-01-01', '2026-01-01', 'new')")
    conn.execute(
        "INSERT INTO number_ledger (video_id, raw_text, normalized, unit, segment_id, start_s, context) "
        "VALUES ('v', '13%', '13', 'pct', 1, 10.0, 'context')")
    row = conn.execute("SELECT crosscheck FROM number_ledger").fetchone()
    assert row["crosscheck"] is None
