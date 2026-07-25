"""Every string here is real transcript text from RagaFinance videos
MgN00MCDDRM and YKnzLYTuVec, or a form the summariser actually emitted.
"""

from __future__ import annotations

import pytest

from ytdigest.numbers import (
    BPS, CLOCK, COUNT, INDEX, MULTIPLE, PCT, SHARES, USD,
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
