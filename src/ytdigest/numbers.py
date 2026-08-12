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
# Bare digits only. A currency quantity must contain one of these, so that
# "千蚊" (colloquial "a thousand-odd") does not become an authoritative
# ledger value of exactly 1000 that a fabricated figure could then match.
_CN_D = "零〇一二兩三四五六七八九"

#: Longest suffix first. "萬" before "千萬" made parse_value's endswith() stop
#: at the wrong boundary, so "5千萬" hit the 萬 branch, left "5千" behind, and
#: parsed to None — the figure could then never verify anything.
MAGNITUDES: list[tuple[str, float]] = [
    ("萬億", 10**12),
    ("千萬", 10**7),
    ("百萬", 10**6),
    ("兆", 10**12),
    ("億", 10**8),
    ("萬", 10**4),
    ("千", 10**3),
    ("百", 10**2),
]


def cn_to_number(text: str) -> float | None:
    """Parse a Chinese numeral string. Returns None if unparseable.

    Handles the compositional forms that actually occur in speech:
    十三 -> 13, 二十三 -> 23, 一百五十 -> 150, 三萬五千 -> 35000.
    """
    text = text.strip()
    if not text or any(c not in _CN_NUM for c in text):
        return None

    # A run of bare digits carries no magnitude word, so it is not
    # compositional and the loop below cannot parse it -- that loop assigns
    # each digit to `current`, so 九九八八 would fall out the bottom as 8.0,
    # the LAST digit, silently. This function used to parse the run instead
    # (七零零 -> 700, reading digits one at a time as a ticker would be
    # spoken). That was itself wrong: the only patterns that fed it real
    # ticker text were later removed from _PATTERNS, because on this corpus
    # a bare digit run overwhelmingly matches Cantonese hesitation and
    # approximation -- a speaker trailing off mid-price -- not a ticker. What
    # still reaches this function is hesitation flowing through the
    # unit-bearing patterns (股/蚊/成/厘), e.g. "二七八蚊" parsing to a clean
    # 278.0 HKD. Refusing outright, rather than parsing accurately, is the
    # only safe choice left: a `None` value comes back `missing` from the
    # validator rather than a confident, fabricated figure.
    if all(c in _DIGITS for c in text) and len(text) >= 2:
        return None

    total = 0.0
    section = 0.0
    current = 0.0
    last_unit: float | None = None
    for char in text:
        if char in _DIGITS:
            current = _DIGITS[char]
        elif char in _SMALL_UNITS:
            unit = _SMALL_UNITS[char]
            # Bare 十 means 10 (十三 = 13), not 0 * 10.
            section += (current or 1) * unit
            current = 0
            last_unit = unit
        elif char in _BIG_UNITS:
            unit = _BIG_UNITS[char]
            section = (section + current) * unit
            total += section
            section = current = 0
            last_unit = unit
        else:
            return None

    # Trailing shorthand: a bare digit after a magnitude means the next
    # magnitude down. 兩萬五 = 25,000 (not 20,005), 三千五 = 3,500, 五百三 = 530.
    # This is the ordinary way index levels and prices are spoken in Cantonese;
    # without it the tail was dropped and the TRUNCATED value was written to
    # the ledger as authoritative, so a summary quoting 20000 for a spoken
    # 兩萬五 verified clean.
    if current and last_unit and last_unit >= 10:
        section += current * (last_unit / 10)
        current = 0

    result = total + section + current
    return result or None


def parse_value(raw: str) -> float | None:
    """Parse either an Arabic or Chinese numeric string, with magnitude suffix."""
    raw = raw.strip().replace(",", "").replace("，", "")
    if not raw:
        return None

    # A pure Chinese numeral is compositional and must be parsed whole.
    # Stripping a magnitude suffix off it double-counts: 五萬三千 became
    # 五萬三 x 1000 = 53,000,000 instead of 53,000, and 二萬八千二百 came out
    # 100x too large. Suffix-stripping exists only for the mixed forms
    # (3千蚊, 5千萬, 39.6666萬億) where the head is Arabic.
    if all(c in _CN_NUM or c in "半幾點" for c in raw):
        return _finish(raw, 1.0)

    multiplier = 1.0
    for suffix, scale in MAGNITUDES:
        if raw.endswith(suffix):
            multiplier = scale
            raw = raw[: -len(suffix)]
            break

    raw = raw.strip()
    if not raw:
        return None
    return _finish(raw, multiplier)


def _finish(raw: str, multiplier: float) -> float | None:

    """Apply 半 / 幾 / 點 handling, then parse."""
    # 半 is half of ONE unit, not half of the magnitude: 三成半 = 3.5 成 = 35%,
    # 三厘半 = 3.5%. This must run before any other parsing, and 半 must not
    # have been swallowed by unit-stripping first — that produced a ledger
    # entry of 30 for a spoken 35%, which a wrong summary then matched.
    half = raw.endswith("半")
    if half:
        raw = raw[:-1].strip()
    # 幾 is an open-ended approximation (兩成幾 = "twenty-odd percent"). Record
    # the floor; it is the only defensible single value.
    raw = raw.removesuffix("幾").strip()
    if not raw:
        return None

    # 點 as a decimal point in a spoken quantity: 三點五 = 3.5.
    if "點" in raw:
        head, _, tail = raw.partition("點")
        head_value = _plain(head)
        if head_value is not None and tail and all(c in _DIGITS for c in tail):
            digits = "".join(str(_DIGITS[c]) for c in tail)
            value = float(f"{head_value:.0f}.{digits}")
            return (value + (0.5 if half else 0.0)) * multiplier

    value = _plain(raw)
    if value is None:
        return None
    if half:
        value += 0.5
    return value * multiplier


def _plain(raw: str) -> float | None:
    """Parse an Arabic or Chinese integer with no suffix handling."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return cn_to_number(raw)


@dataclass
class Found:
    raw: str
    value: float | None
    unit: str
    start: int
    end: int


# A spoken quantity: Arabic, or Chinese with an optional 點 decimal tail.
# Chinese quantities must contain a real digit, so a bare magnitude word
# ("千蚊" = colloquial "a thousand-odd") cannot become an exact ledger value.
_QTY = rf"(?:[\d.,]+|[{_CN_QTY}]*[{_CN_D}][{_CN_QTY}]*(?:點[{_CN_D}]+)?)"
# Magnitude suffixes, longest first so 萬億 beats 萬.
_MAG = r"(?:萬億|千萬|百萬|億|萬|千|百)"

# Order matters: the most specific pattern must win.
_PATTERNS: list[tuple[str, str]] = [
    # 一百五十個基點 / 150個基點 / 150 bps
    (BPS, rf"{_QTY}\s*(?:個)?基點"),
    (BPS, r"[\d.]+\s*(?:bps|BPS|bp)"),
    # 百分之十三 / 百分之5
    (PCT, rf"百分之\s*{_QTY}"),
    # 13% / 13.5%
    (PCT, r"[\d.]+\s*%"),
    # 厘 is how Hong Kong quotes interest rates: 兩厘 = 2%, 4.7厘 = 4.7%,
    # 三厘半 = 3.5%, 三點五厘 = 3.5%. The 半 and 點 forms must be inside the
    # match — dropping them recorded 3 for a spoken 3.5.
    (PCT, rf"{_QTY}\s*厘(?:半)?"),
    # 四成 / 三成半 / 兩成幾  (成 = 10%)
    (PCT, rf"{_QTY}\s*成(?:半|幾)?"),
    # 十五倍 / 15倍 / 15x
    (MULTIPLE, rf"{_QTY}\s*(?:倍|[xX]倍?)"),
    # Clock times must be caught before bare numbers, or 9點半 becomes 9.5.
    # The Chinese branch allows only digits and 十 (十一點半), so that index
    # levels — 三千點, 一萬點 — cannot be swallowed as a time of day and thereby
    # skip validation entirely.
    (CLOCK, rf"(?<![\d.])(?:[1-9]|1[0-2]|[{_CN_D}十]{{1,3}})\s*點(?:半|[\d]+分)?(?![\d{_CN_D}])"),
    # moving averages: 20天線 / 50天線
    (INDEX, r"[\d]+\s*天線"),
    # 2.35億股 / 1.74億股
    (SHARES, rf"{_QTY}\s*{_MAG}?\s*股"),
    # currency with explicit magnitude: 4200億美金 / 二十三億港元 / 3千蚊
    (USD, rf"{_QTY}\s*{_MAG}?\s*(?:美金|美元|USD)"),
    (CNY, rf"{_QTY}\s*{_MAG}?\s*(?:人民幣|人幣|元人民幣|RMB|CNY)"),
    (HKD, rf"{_QTY}\s*{_MAG}?\s*(?:港元|港幣|蚊|HKD)"),
    # Magnitude compounds. The trailing group captures spoken shorthand —
    # 兩萬五 = 25,000, 四萬三 = 43,000 — which was previously truncated to the
    # round number and written to the ledger as authoritative.
    (COUNT, rf"[{_CN_QTY}]*[{_CN_D}][{_CN_QTY}]*{_MAG}[{_CN_QTY}]*"),
    (COUNT, rf"[\d.,]+\s*{_MAG}"),
    # years: 2022年
    (YEAR, r"(?:19|20)\d{2}\s*年"),
    # Bare Arabic numbers, lowest priority so every unit-bearing pattern above
    # wins first. These carry most of the price levels an analyst actually
    # states — 26500, 450, 205 — so omitting them leaves the ledger blind to
    # the most common figure type. Chinese numerals are deliberately NOT
    # matched bare: 一 and 十 occur constantly in ordinary prose.
    (COUNT, r"(?<![\d.,])\d[\d,]*(?:\.\d+)?(?![\d.,])"),
]

_UNIT_STRIP = re.compile(
    r"(個)?基點|bps|BPS|bp|百分之|%|成|厘|倍|[xX]|天線|股|"
    r"美金|美元|USD|人民幣|人幣|RMB|CNY|港元|港幣|蚊|HKD|年"
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
            if unit == CLOCK:
                # Handled apart from the generic path: here 點 is the
                # hour/minute separator, whereas everywhere else it is a
                # decimal point (三點五厘 = 3.5%). Stripping it globally
                # collapsed 三點五 to 三五 and parsed it as 5.
                head = raw.split("點", 1)[0].strip()
                value = _plain(head)
                if value is not None and "半" in raw:
                    value += 0.5
            else:
                body = _UNIT_STRIP.sub("", raw).strip()
                value = parse_value(body)
                if unit == PCT and "成" in raw and value is not None:
                    value *= 10  # 四成 -> 40%
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
