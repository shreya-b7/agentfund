"""Real daily prices from Yahoo Finance via the ``yfinance`` package. No account or keys needed.

Prices are split- and dividend-adjusted (``auto_adjust=True``). yfinance is an
unofficial scraper: it is free and convenient, but it can rate-limit or break
when Yahoo changes its site. It also has little data for delisted companies, so
backtests over past years on today's large caps carry survivorship bias.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .pit import PointInTimeStore
from .universes import LARGE_CAPS


def _download(symbols: list[str], start: str, end: str | None):
    import yfinance as yf  # imported lazily so tests can run without it

    return yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",
        multi_level_index=True,
        progress=False,
        threads=True,
    )


def to_long(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """yfinance's wide (field, ticker) frame -> long ``date,ticker,open,high,low,close,volume``."""
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data (rate limited, offline, or bad tickers)")
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker without a ticker level
        raw = pd.concat({symbols[0]: raw}, axis=1).swaplevel(0, 1, axis=1)
    long = raw.stack(level=1, future_stack=True).reset_index()
    long.columns = [str(c).lower() for c in long.columns]
    long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "ticker"})
    long = long.dropna(subset=["open", "close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long[["date", "ticker", "open", "high", "low", "close", "volume"]]


NY = ZoneInfo("America/New_York")


def drop_unfinished_day(prices: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Remove today's bar if New York's close isn't final yet (yfinance returns live prices intraday)."""
    now = (now or datetime.now(NY)).astimezone(NY)
    if now.time() < time(16, 30):
        prices = prices[prices["date"] < pd.Timestamp(now.date())]
    return prices


def yahoo_store(
    symbols: dict[str, tuple[str, str]] | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    downloader=_download,
) -> PointInTimeStore:
    """Build a :class:`PointInTimeStore` from Yahoo prices. ``symbols``: ticker -> (name, sector)."""
    symbols = symbols or LARGE_CAPS
    tickers = list(symbols)
    end = end or str(date.today() + timedelta(days=1))  # yfinance's end date is exclusive
    prices = drop_unfinished_day(to_long(downloader(tickers, start, end), tickers))
    missing = sorted(set(tickers) - set(prices["ticker"]))
    if missing:  # yfinance's parallel download sometimes drops a ticker ("database is locked"); retry once
        try:
            retry = drop_unfinished_day(to_long(downloader(missing, start, end), missing))
            prices = pd.concat([prices, retry], ignore_index=True)
        except RuntimeError:
            pass
        missing = sorted(set(tickers) - set(prices["ticker"]))
    if missing:
        print(f"warning: no data for {', '.join(missing)}")
    first = prices.groupby("ticker")["date"].min()
    universe = pd.DataFrame(
        [
            {"ticker": t, "name": n, "sector": s, "start_date": first[t], "end_date": pd.NaT}
            for t, (n, s) in symbols.items()
            if t in first.index
        ]
    )
    fundamentals = pd.DataFrame(columns=["ticker", "period_end", "filed_date"])  # EDGAR loader comes later
    return PointInTimeStore(prices=prices, universe=universe, fundamentals=fundamentals)
