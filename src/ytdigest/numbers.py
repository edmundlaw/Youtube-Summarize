"""Chinese/Cantonese numeral parsing and unit classification.

Kept separate from the ledger so it can be tested exhaustively on its own. Every
pattern here came from real transcript text, not from imagination — see
tests/test_numbers.py for the provenance of the awkward ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- units -----------------------------------------------------------------

PCT = "pct"
BPS = "bps"
MULTIPLE = "multiple"
HKD = "hkd"
USD = "usd"
CNY = "cny"
COUNT = "count"
YEAR = "year"
SHARES = "shares"
CLOCK = "clock"  # 9點半 — a time of day, never a financial figure
INDEX = "index"  # 20天線 — a moving average, not a quantity of money

_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_BIG_UNITS = {"萬": 10**4, "億": 10**8, "兆": 10**12}

_CN_NUM = "零〇一二兩三四五六七八九十百千萬億兆"
# Leading quantity may not itself be a magnitude word, or "萬億" parses as a
# figure with no number in front of it.
_CN_QTY = "零〇一二兩三四五六七八九十百千"

# 萬億 must be tried before 萬 and 億 or "39.6666萬億" parses as 39.6666萬.
MAGNITUDES: list[tuple[str, float]] = [
    ("萬億", 10**12),
    ("兆", 10**12),
    ("億", 10**8),
    ("萬", 10**4),
    ("千萬", 10**7),
    ("百萬", 10**6),
]


def cn_to_number(text: str) -> float | None:
    """Parse a Chinese numeral string. Returns None if unparseable.

    Handles the compositional forms that actually occur in speech:
    十三 -> 13, 二十三 -> 23, 一百五十 -> 150, 三萬五千 -> 35000.
    """
    text = text.strip()
    if not text or any(c not in _CN_NUM for c in text):
        return None

    total = 0.0
    section = 0.0
    current = 0.0
    for char in text:
        if char in _DIGITS:
            current = _DIGITS[char]
        elif char in _SMALL_UNITS:
            unit = _SMALL_UNITS[char]
            # Bare 十 means 10 (十三 = 13), not 0 * 10.
            section += (current or 1) * unit
            current = 0
        elif char in _BIG_UNITS:
            section = (section + current) * _BIG_UNITS[char]
            total += section
            section = current = 0
        else:
            return None
    result = total + section + current
    return result or None


def parse_value(raw: str) -> float | None:
    """Parse either an Arabic or Chinese numeric string, with magnitude suffix."""
    raw = raw.strip().replace(",", "").replace("，", "")
    if not raw:
        return None

    multiplier = 1.0
    for suffix, scale in MAGNITUDES:
        if raw.endswith(suffix):
            multiplier = scale
            raw = raw[: -len(suffix)]
            break

    raw = raw.strip()
    if not raw:
        return None

    try:
        return float(raw) * multiplier
    except ValueError:
        pass

    # 三成半 -> 35%, handled by the caller via unit; here 半 is +0.5 of a unit.
    half = raw.endswith("半")
    if half:
        raw = raw[:-1]

    value = cn_to_number(raw)
    if value is None:
        return None
    if half:
        value += 0.5
    return value * multiplier


@dataclass
class Found:
    raw: str
    value: float | None
    unit: str
    start: int
    end: int


# Order matters: the most specific pattern must win.
_PATTERNS: list[tuple[str, str]] = [
    # 一百五十個基點 / 150個基點 / 150 bps
    (BPS, rf"(?:[\d.]+|[{_CN_NUM}]+)\s*(?:個)?基點"),
    (BPS, r"[\d.]+\s*(?:bps|BPS|bp)"),
    # 百分之十三 / 百分之5
    (PCT, rf"百分之\s*(?:[\d.]+|[{_CN_NUM}]+)"),
    # 13% / 13.5%
    (PCT, r"[\d.]+\s*%"),
    # 厘 is how Hong Kong quotes interest rates: 兩厘 = 2%, 4.7厘 = 4.7%.
    # Without this the whole rates discussion lands in the ledger as bare
    # counts, and any summary written as "4.7%" fails to verify.
    (PCT, rf"(?:[\d.]+|[{_CN_QTY}]+)\s*厘"),
    # 四成 / 三成半 / 兩成幾  (成 = 10%)
    (PCT, rf"(?:[\d.]+|[{_CN_NUM}]+)\s*成(?:半|幾)?"),
    # 十五倍 / 15倍 / 15x
    (MULTIPLE, rf"(?:[\d.]+|[{_CN_NUM}]+)\s*(?:倍|[xX]倍?)"),
    # Clock times must be caught before bare numbers, or 9點半 becomes 9.5.
    # Bounded to 1-12 so that "2000點嘅波幅" (index points) is not read as a
    # time of day; the negative lookbehind stops it matching the "0點" inside it.
    (CLOCK, rf"(?<![\d.])(?:[1-9]|1[0-2]|[{_CN_NUM}]{{1,3}})\s*點(?:半|[\d]+分)?(?![\d])"),
    # moving averages: 20天線 / 50天線
    (INDEX, r"[\d]+\s*天線"),
    # 2.35億股 / 1.74億股
    (SHARES, rf"(?:[\d.]+|[{_CN_QTY}]+)\s*(?:萬億|億|萬|千萬|百萬)?\s*股"),
    # currency with explicit magnitude: 4200億美金 / 二十三億港元 / 39.6666萬億
    (USD, rf"(?:[\d.,]+|[{_CN_QTY}]+)\s*(?:萬億|億|萬|千萬|百萬)?\s*(?:美金|美元|USD)"),
    (CNY, rf"(?:[\d.,]+|[{_CN_QTY}]+)\s*(?:萬億|億|萬|千萬|百萬)?\s*(?:人民幣|人幣|元人民幣|RMB|CNY)"),
    (HKD, rf"(?:[\d.,]+|[{_CN_QTY}]+)\s*(?:萬億|億|萬|千萬|百萬)?\s*(?:港元|港幣|蚊|HKD)"),
    # bare magnitude, currency unknown: 4200億 / 16500億 / 39.6666萬億
    (COUNT, rf"(?:[\d.,]+|[{_CN_QTY}]+)\s*(?:萬億|億|萬|千萬|百萬)"),
    # years: 2022年 / 二零二二年
    (YEAR, r"(?:19|20)\d{2}\s*年"),
    # Bare Arabic numbers, lowest priority so every unit-bearing pattern above
    # wins first. These carry most of the price levels an analyst actually
    # states — 26500, 450, 205 — so omitting them leaves the ledger blind to
    # the most common figure type. Chinese numerals are deliberately NOT
    # matched bare: 一 and 十 occur constantly in ordinary prose.
    (COUNT, r"(?<![\d.,])\d[\d,]*(?:\.\d+)?(?![\d.,])"),
]

_UNIT_STRIP = re.compile(
    r"(個)?基點|bps|BPS|bp|百分之|%|成半|成幾|成|厘|倍|[xX]|天線|股|"
    r"美金|美元|USD|人民幣|人幣|RMB|CNY|港元|港幣|蚊|HKD|年|點半|點"
)


def find_numbers(text: str) -> list[Found]:
    """Extract every numeric mention with its unit, longest-match-first.

    Overlapping matches are resolved by preferring the earlier pattern in
    _PATTERNS (more specific) and then the longer span, so that "一百五十個基點"
    yields one bps entry rather than a bare count of 150.
    """
    claimed: list[tuple[int, int]] = []
    found: list[Found] = []

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in claimed)

    for unit, pattern in _PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if overlaps(start, end):
                continue
            raw = match.group(0)
            body = _UNIT_STRIP.sub("", raw).strip()
            value = parse_value(body)
            if unit == PCT:
                if "成" in raw and value is not None:
                    value *= 10  # 四成 -> 40%
            if unit == CLOCK and "半" in raw and value is not None:
                value += 0.5
            claimed.append((start, end))
            found.append(Found(raw=raw, value=value, unit=unit, start=start, end=end))

    return sorted(found, key=lambda f: f.start)


def is_financially_meaningful(unit: str) -> bool:
    """Clock times and moving-average labels are numerals, not figures.

    On one 2.5-hour transcript a naive regex produced 697 'numbers', the
    majority of them 9點半 (half past nine). Passing that to the summariser as
    an authority list is worse than useless — it drowns the real figures.
    """
    return unit not in {CLOCK, INDEX}
