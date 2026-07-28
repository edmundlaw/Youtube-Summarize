-- Daily closes for instruments referenced by views. Local copy: the views
-- table must remain gradeable without a network call, and the household
-- finance database must not be written to by this project.
CREATE TABLE prices (
    symbol   TEXT NOT NULL,
    date     TEXT NOT NULL,          -- YYYY-MM-DD
    open     REAL,
    high     REAL,                   -- needed: a target can be touched intraday
    low      REAL,
    close    REAL NOT NULL,
    source   TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX idx_prices_symbol_date ON prices(symbol, date);
