"""Daily price bars, kept locally so views stay gradeable offline.

Only what the resolver needs: date, open, high, low, close. High and low both
matter — a stated target can be touched intraday and never close there, and
grading on closes alone would mark a reached target as missed.

Nothing here writes to the household finance database; that is a separate
system and this project must not mutate it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .db import now_iso, transaction

#: Our symbol -> the ticker the data source actually uses. Most pass through;
#: these four do not, and each was checked against live data rather than
#: assumed.
YF_SYMBOL: dict[str, str] = {
    "DXY": "DX-Y.NYB",     # dollar index
    "US10Y": "^TNX",       # 10y yield, quoted in percent
    "XAUUSD": "GC=F",      # gold futures stand in for spot
    "USDJPY": "JPY=X",
}


def yf_ticker(symbol: str) -> str:
    return YF_SYMBOL.get(symbol, symbol)


#: Yahoo's chart endpoint, the same one the household finance dashboard uses
#: successfully from this machine. Called directly rather than through
#: yfinance: it is one request, it returns OHLC in one payload, and it removes
#: a dependency that added nothing but a wrapper over this URL.
_CHART = ("https://query1.finance.yahoo.com/v8/finance/chart/"
          "{tkr}?period1={p1}&period2={p2}&interval=1d")
_UA = {"User-Agent": "Mozilla/5.0 (compatible; ytdigest/1.0)"}


def _fetch_one(ticker: str, p1: int, p2: int, attempts: int = 3) -> dict | None:
    """One symbol, retried. Yahoo intermittently drops the TLS connection
    outright (SSL EOF, remote disconnect) rather than returning an error, and a
    single failure would otherwise leave a view permanently ungraded."""
    import json
    import time
    import urllib.request

    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                _CHART.format(tkr=ticker, p1=p1, p2=p2), headers=_UA)
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.loads(response.read())
        except Exception:
            if attempt < attempts:
                time.sleep(2 ** attempt)
    return None


def fetch_prices(conn, symbols: list[str], start: str, end: str) -> dict[str, int]:
    """Download daily bars and upsert. Returns rows stored per symbol.

    A symbol that returns nothing is reported with 0 rather than skipped —
    silently missing price data would show up later as an ungraded view with
    no explanation.
    """
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp())
    p2 = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp())

    stored: dict[str, int] = {}
    for symbol in symbols:
        rows: list[tuple] = []
        payload = _fetch_one(yf_ticker(symbol), p1, p2)
        result = (((payload or {}).get("chart") or {}).get("result") or [None])[0]
        if not result:
            stored[symbol] = 0
            continue
        stamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        for i, stamp in enumerate(stamps):
            close = _num((quote.get("close") or [None] * len(stamps))[i])
            if close is None:
                continue
            rows.append((
                symbol,
                datetime.fromtimestamp(stamp, UTC).date().isoformat(),
                _num((quote.get("open") or [None] * len(stamps))[i]),
                _num((quote.get("high") or [None] * len(stamps))[i]),
                _num((quote.get("low") or [None] * len(stamps))[i]),
                close, "yahoo", now_iso(),
            ))
        if rows:
            with transaction(conn):
                conn.executemany(
                    "INSERT INTO prices (symbol,date,open,high,low,close,source,"
                    "fetched_at) VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(symbol,date) DO UPDATE SET "
                    "  open=excluded.open, high=excluded.high, low=excluded.low, "
                    "  close=excluded.close, fetched_at=excluded.fetched_at",
                    rows,
                )
        stored[symbol] = len(rows)
    return stored


def _num(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None          # NaN -> None


def symbols_needed(conn) -> list[str]:
    """Instruments that views actually reference, so we fetch nothing spare."""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT instrument FROM views WHERE instrument IS NOT NULL "
        "ORDER BY instrument"
    )]


def coverage(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT symbol, COUNT(*) AS bars, MIN(date) AS first, MAX(date) AS last "
        "FROM prices GROUP BY symbol ORDER BY symbol"
    )]


def price_on(conn, symbol: str, date: str) -> float | None:
    """Close on or before `date` — markets are shut at weekends."""
    row = conn.execute(
        "SELECT close FROM prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
        (symbol, date),
    ).fetchone()
    return row["close"] if row else None


def range_high_low(conn, symbol: str, start: str, end: str):
    """(high, low) across a window, or None if there are no bars."""
    row = conn.execute(
        "SELECT MAX(high) AS hi, MIN(low) AS lo FROM prices "
        "WHERE symbol=? AND date>=? AND date<=?", (symbol, start, end),
    ).fetchone()
    if row is None or row["hi"] is None:
        return None
    return row["hi"], row["lo"]


def default_start() -> str:
    """Far enough back to cover any video we would summarise."""
    return f"{datetime.now(UTC).year - 1}-01-01"
