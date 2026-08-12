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
    """`compare`/`_nearest` were dead in production -- `resolve_window`
    superseded both -- and `compare` lacked the `MAX_MISHEARING_EDITS` gate,
    so keeping it around risked a future caller re-inheriting the 25%-
    precision defect that gate fixed. Deleted; these regressions now exercise
    `resolve_window` directly, one caption at a time."""
    from ytdigest.crosscheck import AGREED, DISPUTED, resolve_window

    assert resolve_window([13.0], [13.0]) == [(AGREED, 13.0)]
    assert resolve_window([13.0], [30.0, 56.0])[0][0] == DISPUTED


def test_the_reported_rival_is_the_nearest_one():
    """The digest shows both readings, so it should show the closest competing
    one rather than an unrelated figure from elsewhere in the window."""
    from ytdigest.crosscheck import DISPUTED, resolve_window

    assert resolve_window([13.0], [900.0, 30.0]) == [(DISPUTED, 30.0)]


def test_hearing_no_number_is_absent_not_disputed():
    """Silence, cross-talk or a muffled passage is not evidence the caption is
    wrong. Treating it as disagreement would flag a large share of figures on
    day one and train the reader to ignore the marker."""
    from ytdigest.crosscheck import ABSENT, resolve_window

    assert resolve_window([13.0], []) == [(ABSENT, None)]


def test_a_caption_figure_that_never_parsed_cannot_be_judged():
    from ytdigest.crosscheck import UNCHECKED, resolve_window

    assert resolve_window([None], [13.0]) == [(UNCHECKED, None)]


def test_rounding_noise_is_not_a_dispute():
    """29.9億 written 2990000000.0 against 2990000000 is the same number."""
    from ytdigest.crosscheck import AGREED, resolve_window

    assert resolve_window([2_990_000_000.0], [2_990_000_000.0000001])[0][0] == AGREED


def test_the_real_disagreement_this_was_built_for():
    """MgN00MCDDRM @7483s. Captions 29億, both ASR models 299億, and the
    surrounding figures only make sense at ~300億."""
    from ytdigest.crosscheck import DISPUTED, resolve_window, values_in

    heard = values_in("中芯國際北水淨流入二百九十九億。華虹宏力淨流入五十六億。")
    assert 29_900_000_000 in heard
    assert resolve_window([2_900_000_000], heard) == [(DISPUTED, 29_900_000_000)]


def test_a_dead_asr_leaves_rows_unchecked_and_never_raises(conn, monkeypatch, tmp_path):
    """A safety feature that takes the pipeline down is a worse bug than the one
    it fixes. Every failure path must end with the video still publishable.

    `wav` must be a real path, not None: None short-circuits before the engine is
    ever loaded, so the test would pass without exercising the failure at all.
    """
    from ytdigest import runner

    conn.execute("INSERT INTO channels (id, title, enabled, added_at) "
                 "VALUES ('c', 'Channel', 1, '2026-01-01')")
    conn.execute("INSERT INTO videos (id, channel_id, title, published_at, discovered_at, status) "
                 "VALUES ('v','c','t','2026-01-01','2026-01-01','normalized')")
    conn.execute("INSERT INTO number_ledger (video_id, raw_text, normalized, unit, segment_id, start_s, context) "
                 "VALUES ('v','29億','2900000000','hkd',1,100.0,'')")
    conn.commit()

    loaded = []

    def _boom(cfg):
        loaded.append(True)
        raise RuntimeError("no mlx on this machine")

    monkeypatch.setattr(runner, "_load_asr", _boom)
    cfg = type("C", (), {"get": lambda self, s, k, d=None: d,
                         "data_dir": tmp_path})()
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"")
    counts = runner._crosscheck_figures(cfg, conn, {"id": "v", "duration_s": 600}, wav)

    assert loaded, "engine was never loaded — the failure path was not exercised"
    assert counts == {}
    assert conn.execute("SELECT crosscheck FROM number_ledger").fetchone()[0] is None


def test_a_disabled_crosscheck_is_a_no_op(conn, tmp_path):
    """The config switch must skip the work without touching any row."""
    from ytdigest import runner

    conn.execute("INSERT INTO channels (id, title, enabled, added_at) "
                 "VALUES ('c', 'Channel', 1, '2026-01-01')")
    conn.execute("INSERT INTO videos (id, channel_id, title, published_at, discovered_at, status) "
                 "VALUES ('v','c','t','2026-01-01','2026-01-01','normalized')")
    conn.execute("INSERT INTO number_ledger (video_id, raw_text, normalized, unit, segment_id, start_s, context) "
                 "VALUES ('v','29億','2900000000','hkd',1,100.0,'')")
    conn.commit()

    cfg = type("C", (), {"get": lambda self, s, k, d=None: False,
                         "data_dir": tmp_path})()
    assert runner._crosscheck_figures(cfg, conn, {"id": "v", "duration_s": 600},
                                      tmp_path / "v.wav") == {}


def test_a_window_full_of_figures_does_not_flag_the_innocent_ones():
    """MgN00MCDDRM 7470-7530s, measured. The merged span is 44 seconds and holds
    four different companies' inflows. Judging each caption figure against
    everything heard anywhere in it disputed 56億, 1.7倍 and 76% against numbers
    from other sentences -- 25% precision, which would teach the reader to
    ignore the marker entirely."""
    from ytdigest.crosscheck import AGREED, ABSENT, DISPUTED, resolve_window

    captions = [2_900_000_000, 5_600_000_000, 15_000_000_000, 38_100_000_000,
                30_000_000_000, 5_600_000_000, 1.7, 76]
    heard = [29_900_000_000, 15_000_000_000, 38_100_000_000, 30_000_000_000,
             5_600_000_000]
    got = resolve_window(captions, heard)

    assert got[0] == (DISPUTED, 29_900_000_000)     # the real 10x error
    assert [s for s, _ in got[1:5]] == [AGREED] * 4
    assert [s for s, _ in got[5:]] == [ABSENT] * 3


def test_an_unheard_figure_does_not_dispute_against_a_stranger():
    """A leftover reading may only dispute a figure it plausibly IS. 76 against
    5,600,000,000 is nine digit edits apart -- different figures, not a
    mishearing."""
    from ytdigest.crosscheck import ABSENT, DISPUTED, resolve_window

    assert resolve_window([76], [5_600_000_000]) == [(ABSENT, None)]
    assert resolve_window([13], [30]) == [(DISPUTED, 30)]


def test_one_heard_reading_vouches_for_only_one_caption_figure():
    """The speaker says 56億 twice; ASR caught it once. We can honestly vouch for
    one of them."""
    from ytdigest.crosscheck import AGREED, ABSENT, resolve_window

    assert resolve_window([5_600_000_000, 5_600_000_000], [5_600_000_000]) == [
        (AGREED, 5_600_000_000), (ABSENT, None)]


def test_the_dispute_lands_on_the_likelier_mishearing():
    """captions 13 and 31 both compete for a single heard 30. 31 is one digit
    edit away and 13 is two, so 31 is what 30 most plausibly mis-hears. Greedy
    caption-order assignment gave 30 to whichever appeared first in an unordered
    SELECT -- so which figure carried the warning was arbitrary."""
    from ytdigest.crosscheck import ABSENT, DISPUTED, resolve_window

    assert resolve_window([13, 31], [30]) == [(ABSENT, None), (DISPUTED, 30)]
    assert resolve_window([31, 13], [30]) == [(DISPUTED, 30), (ABSENT, None)]


def test_wants_crosscheck_is_false_once_every_row_is_judged(conn):
    """Unlike `_wants_voice`, which checks whether identification already ran,
    `_wants_crosscheck` used to want the work whenever any row simply had a
    `start_s`, checked or not -- so every re-run of `stage_identify` on an
    already-judged video re-downloaded ~570 MB and re-paid 10+ minutes of ASR
    on rows that already carried a verdict."""
    from ytdigest import runner

    conn.execute("INSERT INTO channels (id, title, enabled, added_at) "
                 "VALUES ('c', 'Channel', 1, '2026-01-01')")
    conn.execute("INSERT INTO videos (id, channel_id, title, published_at, discovered_at, status) "
                 "VALUES ('v','c','t','2026-01-01','2026-01-01','normalized')")
    conn.execute(
        "INSERT INTO number_ledger (video_id, raw_text, normalized, unit, segment_id, "
        "start_s, context, crosscheck) VALUES ('v','29億','2900000000','hkd',1,100.0,'','agreed')")
    conn.commit()

    cfg = type("C", (), {"get": lambda self, s, k, d=None: True})()
    assert runner._wants_crosscheck(cfg, conn, {"id": "v"}) is False


def test_wants_crosscheck_is_true_while_a_row_is_still_unjudged(conn):
    from ytdigest import runner

    conn.execute("INSERT INTO channels (id, title, enabled, added_at) "
                 "VALUES ('c', 'Channel', 1, '2026-01-01')")
    conn.execute("INSERT INTO videos (id, channel_id, title, published_at, discovered_at, status) "
                 "VALUES ('v','c','t','2026-01-01','2026-01-01','normalized')")
    conn.execute(
        "INSERT INTO number_ledger (video_id, raw_text, normalized, unit, segment_id, "
        "start_s, context) VALUES ('v','29億','2900000000','hkd',1,100.0,'')")
    conn.commit()

    cfg = type("C", (), {"get": lambda self, s, k, d=None: True})()
    assert runner._wants_crosscheck(cfg, conn, {"id": "v"}) is True


def test_no_wav_download_when_neither_voice_nor_crosscheck_has_work(conn, monkeypatch, tmp_path):
    """Before a channel's first voiceprint is enrolled, `_identify_speakers`
    returns immediately -- so with the cross-check also off, nothing should
    ever open `audio_for`. Downloading anyway costs ~570 MB per video, thrown
    away unused, on every single video before that first enrolment."""
    import json

    from ytdigest import runner
    from ytdigest import voice

    conn.execute("INSERT INTO channels (id, title, enabled, added_at) "
                 "VALUES ('c', 'Channel', 1, '2026-01-01')")
    conn.execute("INSERT INTO videos (id, channel_id, title, published_at, discovered_at, status) "
                 "VALUES ('v','c','t','2026-01-01','2026-01-01','normalized')")
    normalized_path = tmp_path / "v.json"
    normalized_path.write_text(json.dumps({"segments": []}), encoding="utf-8")
    conn.execute(
        "INSERT INTO artifacts (video_id, kind, path, sha256, bytes, created_at) "
        "VALUES ('v','normalized',?,?,?,?)",
        (str(normalized_path), "x" * 8, normalized_path.stat().st_size, "2026-01-01"))
    conn.commit()

    def _boom(video_id, dest_dir):
        raise AssertionError("audio_for should not be called")

    monkeypatch.setattr(voice, "audio_for", _boom)

    # voice.enabled=True (so _identify_speakers's own gate isn't what stops
    # it -- the no-voiceprints early return is), asr.crosscheck=False.
    def _get(self, section, key, default=None):
        if (section, key) == ("voice", "enabled"):
            return True
        return False

    cfg = type("C", (), {"get": _get, "data_dir": tmp_path})()

    result = runner.stage_identify(cfg, conn, {"id": "v", "duration_s": 600})
    assert result is None
