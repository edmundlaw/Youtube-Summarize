"""Every string here is real transcript text from RagaFinance videos
MgN00MCDDRM and YKnzLYTuVec, or a form the summariser actually emitted.
"""

from __future__ import annotations

import pytest

from ytdigest.numbers import (
    BPS, CLOCK, COUNT, HKD, INDEX, MULTIPLE, PCT, SHARES, USD,
    cn_to_number, find_numbers, is_financially_meaningful, parse_value,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("十三", 13), ("二十三", 23), ("一百五十", 150), ("三", 3),
        ("十", 10), ("二十", 20), ("三萬五千", 35000), ("四", 4),
        ("兩", 2), ("一億", 10**8),
    ],
)
def test_chinese_numerals(text, expected):
    assert cn_to_number(text) == expected


def test_magnitude_suffixes():
    assert parse_value("4200億") == 4200 * 10**8
    assert parse_value("2.35億") == 2.35 * 10**8
    # 萬億 must beat 萬 and 億 individually, or this parses as 39.6666 * 10^4.
    assert parse_value("39.6666萬億") == pytest.approx(39.6666 * 10**12)


def _units(text):
    return {(f.raw, f.unit) for f in find_numbers(text)}


def test_percent_forms():
    assert ("百分之十三", PCT) in _units("free cash flow上季升咗百分之十三")
    assert ("7.1%", PCT) in _units("騰訊股價急跌7.1%")


def test_seng_is_ten_percent():
    """四成 = 40%. The summariser normalises it silently, so the ledger must
    hold the same normalised value or the validator will reject a correct figure."""
    found = [f for f in find_numbers("payout ratio維持喺四成左右") if f.unit == PCT]
    assert found and found[0].value == 40


def test_basis_points():
    found = [f for f in find_numbers("operating margin跌咗一百五十個基點") if f.unit == BPS]
    assert found and found[0].value == 150


def test_multiple():
    found = [f for f in find_numbers("forward P/E大概十五倍") if f.unit == MULTIPLE]
    assert found and found[0].value == 15
    assert any(f.unit == MULTIPLE and f.value == 8 for f in find_numbers("吸升咗8倍咁多"))


def test_currency_with_magnitude():
    found = [f for f in find_numbers("淨係表債務已經有4200億美金") if f.unit == USD]
    assert found and found[0].value == 4200 * 10**8


def test_bare_magnitude_is_count_not_currency():
    """'16500億' with no currency word must not be guessed as USD."""
    found = [f for f in find_numbers("隱藏住嘅債務達到係16500億") if f.unit == COUNT]
    assert found and found[0].value == 16500 * 10**8


def test_shares():
    found = [f for f in find_numbers("第一季嘅2.35億股") if f.unit == SHARES]
    assert found and found[0].value == 2.35 * 10**8


def test_clock_time_is_not_a_figure():
    """9點半 appeared ~100 times in one transcript. Treating it as a financial
    figure floods the ledger and buries the real numbers."""
    found = [f for f in find_numbers("每個交易日早上9點半開始") if f.unit == CLOCK]
    assert found
    assert not is_financially_meaningful(CLOCK)


def test_moving_average_label_is_not_a_figure():
    found = [f for f in find_numbers("穿咗450蚊，個20天線同50天線") if f.unit == INDEX]
    assert found
    assert not is_financially_meaningful(INDEX)


def test_hkd_colloquial_shing():
    """蚊 is the spoken HK dollar. 450蚊 is a price, not a count."""
    from ytdigest.numbers import HKD

    assert any(f.unit == HKD and f.value == 450 for f in find_numbers("穿咗450蚊"))


def test_specific_pattern_wins_over_generic():
    """一百五十個基點 must yield one bps entry, not a stray count of 150."""
    found = find_numbers("跌咗一百五十個基點")
    assert len([f for f in found if f.value == 150]) == 1
    assert found[0].unit == BPS


def test_no_spurious_matches_on_prose():
    assert find_numbers("大家好，今日冇乜特別") == []


def test_lei_is_percent_for_hk_rates():
    """厘 is how Hong Kong quotes interest rates: 兩厘 = 2%. Parsing these as
    bare counts made every rates figure unverifiable."""
    found = [f for f in find_numbers("10年債息由4厘推到4.7厘") if f.unit == PCT]
    assert {f.value for f in found} == {4.0, 4.7}
    assert any(f.unit == PCT and f.value == 2 for f in find_numbers("可能要加到兩厘"))


# --- regressions from the 2026-07-25 cloud review -------------------------
# Each of these produced a FALSE PASS: the ledger recorded a truncated or
# wrong value, so a summary quoting that wrong value verified clean.

def test_half_unit_is_not_truncated():
    """三成半 = 35%, not 30%. 半 was stripped before it could be parsed."""
    assert any(f.unit == PCT and f.value == 35 for f in find_numbers("毛利率有三成半"))
    assert any(f.unit == PCT and f.value == 3.5 for f in find_numbers("息率三厘半"))


def test_trailing_shorthand_is_not_truncated():
    """兩萬五 = 25,000 — the ordinary way an index level is spoken. Recording
    20,000 let a summary saying 20000 verify against a spoken 25,000."""
    assert any(f.value == 25000 for f in find_numbers("恒指兩萬五"))
    assert any(f.value == 43000 for f in find_numbers("四萬三"))
    assert cn_to_number("三千五") == 3500
    assert cn_to_number("五百三") == 530
    assert cn_to_number("二十三") == 23      # unchanged
    assert cn_to_number("三萬五千") == 35000  # unchanged


def test_chinese_decimal_point():
    """三點五厘 = 3.5%. 點 was read as a clock separator, giving 5."""
    assert any(f.unit == PCT and f.value == 3.5 for f in find_numbers("加到三點五厘"))
    assert any(f.unit == MULTIPLE and f.value == 3.5 for f in find_numbers("三點五倍"))


def test_clock_still_works():
    assert any(f.unit == CLOCK and f.value == 9.5 for f in find_numbers("9點半開始"))
    assert any(f.unit == CLOCK and f.value == 11.5 for f in find_numbers("十一點半開市"))


def test_index_level_is_not_swallowed_as_a_clock_time():
    """三千點 landed in CLOCK, and check_text skips CLOCK entirely — so a
    fabricated index target written in Chinese numerals escaped validation."""
    units = {f.unit for f in find_numbers("恒指目標三千點")}
    assert CLOCK not in units
    assert any(f.value == 3000 for f in find_numbers("恒指目標三千點"))


def test_magnitude_suffix_ordering():
    """MAGNITUDES had 萬 before 千萬, so 5千萬 parsed to None and could never
    verify anything."""
    assert any(f.value == 5 * 10**7 for f in find_numbers("派咗5千萬"))
    assert any(f.value == 8 * 10**6 for f in find_numbers("市值8百萬"))
    assert any(f.value == 3000 for f in find_numbers("每手3千蚊"))


def test_bare_magnitude_word_is_not_a_currency_figure():
    """千蚊 colloquially means 'a thousand-odd'. Recording an exact 1000 gave
    a fabricated 1000蚊 something to match against."""
    assert not [f for f in find_numbers("每手要千蚊") if f.unit == HKD]


def test_compound_chinese_magnitudes_are_not_multiplied_twice():
    """Regression: magnitude-suffix stripping fought the compositional parser.
    五萬三千 (53,000) came out 53,000,000 and 二萬八千二百 (28,200) came out
    100x too large — a wrong value written to the ledger as authoritative."""
    assert parse_value("五萬三千") == 53000
    assert parse_value("二萬八千二百") == 28200
    assert parse_value("三萬五千") == 35000
    # the mixed Arabic+Chinese forms that suffix-stripping exists for
    assert parse_value("5千萬") == 5 * 10**7
    assert parse_value("39.6666萬億") == pytest.approx(39.6666 * 10**12)


def test_figures_do_not_span_flattened_fields():
    """Regression: patterns used \\s*, which crossed the newline joining two
    unrelated summary fields — a figure ending one field joined the 股 opening
    the next, producing phantom '165\\n股' entries that could never verify."""
    from ytdigest.summarize import _flatten

    flat = _flatten({"numbers": [{"figure": "165", "context": ""}],
                     "theses": [{"thesis": "股份分析", "reasoning": ""}]})
    assert not [f for f in find_numbers(flat) if "\n" in f.raw]


def test_spoken_digit_strings_are_not_read_as_their_last_digit():
    """九九八八 is Alibaba's ticker 9988, spoken digit by digit. The
    compositional parser overwrote its accumulator on each digit and returned
    8.0 -- silently wrong, which is worse than refusing. Qwen3-ASR renders every
    ticker this way, so without this the cross-check compares noise."""
    from ytdigest.numbers import cn_to_number

    assert cn_to_number("九九八八") == 9988
    assert cn_to_number("七零零") == 700
    assert cn_to_number("一三四七") == 1347
    assert cn_to_number("九八一") == 981


def test_two_digit_runs_are_refused_as_ambiguous():
    """兩三 is "two or three" -- an approximation, not 23. Guessing here would
    invent a figure, which is the one failure this project does not accept."""
    from ytdigest.numbers import cn_to_number

    assert cn_to_number("兩三") is None
    assert cn_to_number("三四") is None
    assert cn_to_number("五六") is None


def test_compositional_numerals_are_unaffected():
    """The forms that already worked must not regress: these carry every real
    figure in the corpus."""
    from ytdigest.numbers import cn_to_number

    assert cn_to_number("二百九十九") == 299
    assert cn_to_number("十三") == 13
    assert cn_to_number("三萬五千") == 35000
    assert cn_to_number("五") == 5


def test_spoken_tickers_enter_the_ledger():
    """YouTube's captions write tickers as Arabic digits and Qwen speaks them.
    Unless both reach the ledger there is nothing to cross-check."""
    from ytdigest.numbers import find_numbers

    got = {f.raw: f.value for f in find_numbers("譬如七零零啦，九九八八啦，一三四七啦")}
    assert got["七零零"] == 700
    assert got["九九八八"] == 9988
    assert got["一三四七"] == 1347


def test_spoken_years_are_years_not_quantities():
    """二零二五年 must classify as YEAR like its Arabic twin, or is_financially
    _meaningful lets a calendar year into the ledger as a figure."""
    from ytdigest.numbers import YEAR, find_numbers

    got = [f for f in find_numbers("到二零二五年為止") if f.unit == YEAR]
    assert got and got[0].value == 2025


def test_ordinary_prose_still_yields_no_bare_chinese_numbers():
    """一 and 十 are everywhere in speech. Only runs of three or more count."""
    from ytdigest.numbers import find_numbers

    assert find_numbers("我一於唔買，十分之危險") == []
