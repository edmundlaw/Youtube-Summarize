"""Price storage. The resolver depends on high/low being present, so these
tests guard the shape of the data rather than the network."""

from __future__ import annotations

import pathlib
import tempfile

from ytdigest import db as D
from ytdigest.prices import YF_SYMBOL, coverage, price_on, range_high_low, yf_ticker

SYMBOLS = ["0005.HK", "000660.KS", "0700.HK", "3690.HK", "AAPL", "BTC-USD",
           "CL=F", "DXY", "GOOGL", "TSLA", "US10Y", "USDJPY", "XAUUSD",
           "^DJI", "^GSPC", "^HSI", "^IXIC", "^KS11"]


def _conn():
    return D.open_db(pathlib.Path(tempfile.mkdtemp()) / "t.db",
                     pathlib.Path("migrations"))


def test_every_symbol_resolves_to_a_ticker():
    for symbol in SYMBOLS:
        assert yf_ticker(symbol)


def test_only_the_four_known_remaps_exist():
    """A silent remap is how the wrong instrument's prices get graded against
    someone's call."""
    assert set(YF_SYMBOL) == {"DXY", "US10Y", "XAUUSD", "USDJPY"}


def test_upsert_does_not_duplicate():
    conn = _conn()
    for _ in range(2):
        conn.execute(
            "INSERT INTO prices (symbol,date,open,high,low,close,source,fetched_at) "
            "VALUES ('^HSI','2026-01-05',1,2,0.5,1.5,'test',?) "
            "ON CONFLICT(symbol,date) DO UPDATE SET close=excluded.close",
            (D.now_iso(),))
    assert coverage(conn)[0]["bars"] == 1


def test_price_on_falls_back_to_the_previous_session():
    """Markets are shut at weekends; asking for Sunday must not return None."""
    conn = _conn()
    conn.execute("INSERT INTO prices (symbol,date,open,high,low,close,source,"
                 "fetched_at) VALUES ('^HSI','2026-01-02',1,2,0.5,1.5,'t',?)",
                 (D.now_iso(),))
    assert price_on(conn, "^HSI", "2026-01-04") == 1.5
    assert price_on(conn, "^HSI", "2026-01-01") is None


def test_range_high_low_returns_none_for_unknown_symbol():
    assert range_high_low(_conn(), "NOSUCH", "2026-01-01", "2026-12-31") is None
