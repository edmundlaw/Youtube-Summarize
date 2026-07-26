"""Regression tests built from three failures observed in live runs.

If any of these ever go green-to-red, a real bad number is reaching the user.
"""

from __future__ import annotations

from ytdigest.interfaces import Segment
from ytdigest.normalize import build_ledger
from ytdigest.validator import (
    FAILED, PASSED, PASSED_WITH_FLAGS,
    annotate, check_text, offending, retry_instruction, verdict,
)


def _ledger(*texts: str, confidence: float | None = None):
    segs = [
        Segment(id=i, start=float(i * 10), end=float(i * 10 + 9), text=t,
                confidence=confidence)
        for i, t in enumerate(texts)
    ]
    return build_ledger(segs)


def test_fabricated_figure_is_caught():
    """The 28243 case: captions truncated, model invented an HSI target."""
    ledger = _ledger("下一個目標應該係26500度啦，26000又會錯，所以應該係八、二")
    checks = check_text("恒指目標28243", ledger)
    assert verdict(checks) == FAILED
    assert "28243" in offending(checks)


def test_figure_actually_present_passes():
    ledger = _ledger("下一個目標應該係26500度啦")
    checks = check_text("目標26500", ledger)
    assert verdict(checks) == PASSED
    assert offending(checks) == []


def test_unit_stripping_does_not_create_a_false_pass():
    """The 2000億 case: a bare '2000' elsewhere must not satisfy '2000億'."""
    ledger = _ledger("大概2000點嘅波幅", "另外講到 free cash flow 轉負")
    checks = check_text("capex達2000億", ledger)
    assert verdict(checks) == FAILED
    assert any("2000億" in f for f in offending(checks))


def test_split_number_is_flagged_not_failed():
    """The 4200|億美金 case: transcript corruption separated value from unit.
    The figure is real, so reject-outright would punish a correct summary."""
    ledger = _ledger("淨係表債務已經有4200億")          # bare magnitude, no currency
    checks = check_text("Meta表外債務4200億美金", ledger)
    assert verdict(checks) == PASSED_WITH_FLAGS
    assert offending(checks) == []
    assert any("unit differs" in c.reason for c in checks)


def test_chinese_numeral_in_source_matches_arabic_in_summary():
    """Models silently normalise 百分之十三 -> 13%. The ledger stores the
    normalised value, so the match must still succeed."""
    ledger = _ledger("free cash flow上季升咗百分之十三")
    assert verdict(check_text("FCF升13%", ledger)) == PASSED


def test_seng_normalisation_matches():
    ledger = _ledger("payout ratio維持喺四成左右")
    assert verdict(check_text("payout ratio約40%", ledger)) == PASSED


def test_low_confidence_source_is_always_flagged():
    """Even a correct figure must be marked when its source segment is shaky."""
    ledger = _ledger("升咗百分之十三", confidence=0.3)
    checks = check_text("升13%", ledger)
    assert verdict(checks) == PASSED_WITH_FLAGS


def test_clock_times_are_not_validated_as_figures():
    """9點半 is not a financial figure and must not need a ledger entry."""
    ledger = _ledger("大市今日回落")
    checks = check_text("每個交易日9點半開始", ledger)
    assert verdict(checks) == PASSED


def test_annotate_marks_only_unverified():
    ledger = _ledger("目標係26500度")
    checks = check_text("由26500升到28243", ledger)
    out = annotate("由26500升到28243", checks)
    assert "⚠︎28243" in out
    assert "⚠︎26500" not in out


def test_retry_instruction_names_the_offender_and_lists_allowed():
    ledger = _ledger("目標係26500度啦")
    checks = check_text("目標28243", ledger)
    instruction = retry_instruction(checks, ledger)
    assert "28243" in instruction
    assert "26500" in instruction
    assert "字幕於此中斷" in instruction


def test_no_numbers_is_a_pass():
    assert verdict(check_text("大市今日冇乜方向", _ledger("大市回落"))) == PASSED


def test_chinese_magnitude_in_source_matches_arabic_in_summary():
    """Regression: the real transcript said 二千億; a string-matching check
    searched for '2000億', found nothing, and wrongly reported a fabrication.
    Comparison must be on normalised value, never on text."""
    ledger = _ledger("我哋資本開始去到二千億")
    assert verdict(check_text("capex達2000億", ledger)) == PASSED


def test_colloquial_currency_flags_rather_than_fails():
    """Cantonese 蚊 is used for dollars generically. A US-listed stock quoted
    as 223蚊 is correctly rendered 223美元 — flag it, do not cry fabrication."""
    ledger = _ledger("Alp二百。223蚊啦，嗰啲就抵買啲")
    checks = check_text("Alphabet可能跌至223美元", ledger)
    assert verdict(checks) == PASSED_WITH_FLAGS
    assert offending(checks) == []


def test_lenient_json_repairs_unescaped_inner_quote():
    """Observed live: the model quoted transcript text inside a value without
    escaping, breaking the whole response. One stray quote must not cost a
    2.5-hour summary."""
    from ytdigest.summarize import loads_lenient

    broken = '{"numbers": [{"figure": "202", "context": "字幕寫"202" 唔清楚", "ts": "06:04"}]}'
    parsed = loads_lenient(broken)
    assert parsed["numbers"][0]["figure"] == "202"
    assert "202" in parsed["numbers"][0]["context"]


def test_lenient_json_leaves_valid_json_alone():
    from ytdigest.summarize import loads_lenient

    assert loads_lenient('{"a": ["x", "y"], "b": {"c": 1}}') == {"a": ["x", "y"], "b": {"c": 1}}


def test_annotation_does_not_corrupt_other_figures():
    """Regression: marking {1200億, 200億} produced ⚠︎1⚠︎200億, and {13%, 3%}
    produced ⚠︎1⚠︎3% — the safety annotation manufacturing a wrong number."""
    from ytdigest.publish import _annotate

    from ytdigest.validator import Check

    def marks(figures):
        return [Check(figure=f, unit="count", value=None,
                      verdict="missing", reason="x") for f in figures]

    assert _annotate("收入1200億，成本200億", marks(["1200億", "200億"])) == \
        "收入⚠︎1200億，成本⚠︎200億"
    assert _annotate("毛利率 13%，股息率 3%", marks(["13%", "3%"])) == \
        "毛利率 ⚠︎13%，股息率 ⚠︎3%"


def test_killed_stage_eventually_abandons():
    """Regression: a killed stage left a 'running' row, so attempts_for stayed
    0 — no backoff, no abandonment, full-cost re-run every scheduled invocation
    forever."""
    import pathlib
    import tempfile

    from ytdigest import db as D

    db_path = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    conn = D.open_db(db_path, pathlib.Path("migrations"))
    conn.execute("INSERT INTO channels (id,title,added_at) VALUES ('UC1','c',?)",
                 (D.now_iso(),))
    conn.execute(
        "INSERT INTO videos (id,channel_id,title,published_at,discovered_at,status)"
        " VALUES ('v1','UC1','t','2026-01-01','2026-01-01',?)", (D.NORMALIZED,))

    for _ in range(3):
        D.start_stage(conn, "v1", "summarize",
                      D.attempts_for(conn, "v1", "summarize") + 1)
        conn.execute("UPDATE stage_runs SET started_at='2020-01-01T00:00:00+00:00'"
                     " WHERE status='running'")
        assert D.reap_orphan_runs(conn, 900) == 1

    assert D.attempts_for(conn, "v1", "summarize") == 3
    status = conn.execute("SELECT status FROM videos WHERE id='v1'").fetchone()["status"]
    assert status == D.ABANDONED
    assert D.claim_queue(conn, 10) == []
