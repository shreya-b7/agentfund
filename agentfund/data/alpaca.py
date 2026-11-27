"""Real daily prices from Alpaca's free market data API.

The free plan serves full-market (SIP) historical bars as long as they are more
than 15 minutes old, so daily bars up to yesterday's close are fine. History
starts around 2016. Prices are split- and dividend-adjusted.

Caveat: the default universe is today's large caps, so a backtest over past
years has survivorship bias (it never holds a company that later collapsed).
That's fine for paper trading today; for research results use a source with
delisted names.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd

from ..broker.alpaca import DATA_URL, _headers, _keys
from .pit import PointInTimeStore
from .universes import LARGE_CAPS


def fetch_daily_bars(
    symbols: list[str],
    start: str,
    end: str | None = None,
    key_id: str | None = None,
    secret: str | None = None,
    transport: httpx.BaseTransport | None = None,
    batch: int = 50,
) -> pd.DataFrame:
    end = end or str(date.today() - timedelta(days=1))
    kid, sec = _keys(key_id, secret)
    rows = []
    with httpx.Client(base_url=DATA_URL, headers=_headers(kid, sec), timeout=60, transport=transport) as http:
        for i in range(0, len(symbols), batch):
            chunk = symbols[i : i + batch]
            token = None
            while True:
                params = {
                    "symbols": ",".join(chunk),
                    "timeframe": "1Day",
                    "start": start,
                    "end": end,
                    "adjustment": "all",
                    "feed": "sip",
                    "limit": 10000,
                }
                if token:
                    params["page_token"] = token
                r = http.get("/v2/stocks/bars", params=params)
                r.raise_for_status()
                data = r.json()
                for sym, bars in (data.get("bars") or {}).items():
                    for b in bars:
                        rows.append((b["t"], sym, b["o"], b["h"], b["l"], b["c"], b["v"]))
                token = data.get("next_page_token")
                if not token:
                    break
    df = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.drop_duplicates(["date", "ticker"])


def alpaca_store(
    symbols: dict[str, tuple[str, str]] | None = None,
    start: str = "2016-01-01",
    end: str | None = None,
    **kw,
) -> PointInTimeStore:
    """Build a :class:`PointInTimeStore` from Alpaca bars. ``symbols``: ticker -> (name, sector)."""
    symbols = symbols or LARGE_CAPS
    prices = fetch_daily_bars(list(symbols), start, end, **kw)
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
