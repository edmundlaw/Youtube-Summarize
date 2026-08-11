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


def test_spans_merge_and_clamp():
    """Figures cluster. Overlapping windows must merge or the same seconds get
    transcribed several times -- measured at 29% of audio when merged."""
    from ytdigest.crosscheck import spans_for

    assert spans_for([100.0, 104.0, 500.0], duration_s=600.0, window_s=8.0) == [
        (92.0, 112.0), (492.0, 508.0)]


def test_spans_never_run_past_either_end():
    from ytdigest.crosscheck import spans_for

    assert spans_for([3.0], duration_s=10.0, window_s=8.0) == [(0.0, 10.0)]


def test_agreement_needs_the_same_value():
    from ytdigest.crosscheck import AGREED, DISPUTED, compare

    assert compare(13.0, [13.0]) == (AGREED, 13.0)
    assert compare(13.0, [30.0, 56.0])[0] == DISPUTED


def test_the_reported_rival_is_the_nearest_one():
    """The digest shows both readings, so it should show the closest competing
    one rather than an unrelated figure from elsewhere in the window."""
    from ytdigest.crosscheck import DISPUTED, compare

    assert compare(13.0, [900.0, 30.0]) == (DISPUTED, 30.0)


def test_hearing_no_number_is_absent_not_disputed():
    """Silence, cross-talk or a muffled passage is not evidence the caption is
    wrong. Treating it as disagreement would flag a large share of figures on
    day one and train the reader to ignore the marker."""
    from ytdigest.crosscheck import ABSENT, compare

    assert compare(13.0, []) == (ABSENT, None)


def test_a_caption_figure_that_never_parsed_cannot_be_judged():
    from ytdigest.crosscheck import UNCHECKED, compare

    assert compare(None, [13.0]) == (UNCHECKED, None)


def test_rounding_noise_is_not_a_dispute():
    """29.9億 written 2990000000.0 against 2990000000 is the same number."""
    from ytdigest.crosscheck import AGREED, compare

    assert compare(2_990_000_000.0, [2_990_000_000.0000001])[0] == AGREED


def test_the_real_disagreement_this_was_built_for():
    """MgN00MCDDRM @7483s. Captions 29億, both ASR models 299億, and the
    surrounding figures only make sense at ~300億."""
    from ytdigest.crosscheck import DISPUTED, compare, values_in

    heard = values_in("中芯國際北水淨流入二百九十九億。華虹宏力淨流入五十六億。")
    assert 29_900_000_000 in heard
    assert compare(2_900_000_000, heard) == (DISPUTED, 29_900_000_000)
