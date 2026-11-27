"""Deterministic synthetic market for offline development and tests.

This is not real data. It exists so the whole pipeline runs (and CI passes)
without network access or data licences. It deliberately includes the things
that break naive backtests: listings mid-sample, delistings after crashes,
reporting lags and restated fundamentals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pit import PointInTimeStore

SECTORS = ["Tech", "Health", "Financials", "Energy", "Consumer", "Industrials"]


def generate_market(
    n_tickers: int = 60,
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    seed: int = 7,
) -> PointInTimeStore:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days = len(dates)

    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    sectors = [SECTORS[i % len(SECTORS)] for i in range(n_tickers)]
    sector_idx = np.array([SECTORS.index(s) for s in sectors])

    # Factor model: market + sector + idiosyncratic, with slowly drifting
    # per-name expected returns so that momentum carries some (weak) signal.
    mkt = rng.normal(0.0003, 0.010, n_days)
    sec = rng.normal(0.0, 0.006, (n_days, len(SECTORS)))
    beta = rng.uniform(0.6, 1.4, n_tickers)
    idio_vol = rng.uniform(0.010, 0.025, n_tickers)
    drift = np.zeros((n_days, n_tickers))
    d = rng.normal(0, 0.0004, n_tickers)
    for t in range(n_days):
        d = 0.995 * d + rng.normal(0, 0.00004, n_tickers)
        drift[t] = d
    rets = mkt[:, None] * beta + sec[:, sector_idx] + drift + rng.normal(0, 1, (n_days, n_tickers)) * idio_vol

    # Listing / delisting schedule.
    start_idx = np.zeros(n_tickers, dtype=int)
    late = rng.choice(n_tickers, size=n_tickers // 6, replace=False)
    start_idx[late] = rng.integers(60, n_days // 2, size=len(late))
    end_idx = np.full(n_tickers, n_days)  # exclusive
    doomed = rng.choice(np.setdiff1d(np.arange(n_tickers), late), size=n_tickers // 8, replace=False)
    for i in doomed:
        e = int(rng.integers(n_days // 3, n_days - 20))
        rets[e - 40 : e, i] -= 0.012  # slide before delisting
        end_idx[i] = e

    rows = []
    for i, tk in enumerate(tickers):
        s, e = start_idx[i], end_idx[i]
        r = rets[s:e, i]
        close = 20.0 * rng.uniform(0.5, 5.0) * np.exp(np.cumsum(r))
        gap = rng.normal(0, 0.003, len(r))
        open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(gap)
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(r))))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(r))))
        vol = (rng.lognormal(13.5, 0.4, len(r)) * (1 + 20 * np.abs(r))).round()
        rows.append(
            pd.DataFrame(
                {"date": dates[s:e], "ticker": tk, "open": open_, "high": hi, "low": lo, "close": close, "volume": vol}
            )
        )
    prices = pd.concat(rows, ignore_index=True)

    universe = pd.DataFrame(
        {
            "ticker": tickers,
            "name": [f"Synthetic Co {i:03d}" for i in range(n_tickers)],
            "sector": sectors,
            "start_date": dates[start_idx],
            "end_date": [dates[e] if e < n_days else pd.NaT for e in end_idx],
        }
    )

    # Quarterly fundamentals filed 25-55 days after quarter end; ~10% restated later.
    frows = []
    quarter_ends = pd.date_range(start, end, freq="QE")
    for i, tk in enumerate(tickers):
        listed_from, listed_to = dates[start_idx[i]], dates[min(end_idx[i], n_days) - 1]
        revenue = rng.uniform(200, 5000)
        for q in quarter_ends:
            if q < listed_from or q > listed_to:
                continue
            revenue *= np.exp(rng.normal(0.01 + 40 * drift[dates.searchsorted(q) - 1, i], 0.05))
            margin = rng.normal(0.12, 0.05)
            filed = q + pd.Timedelta(days=int(rng.integers(25, 56)))
            if filed > listed_to:
                continue
            eps = revenue * margin / 100.0
            frows.append((tk, q, filed, revenue, eps))
            if rng.random() < 0.10:
                restated = filed + pd.Timedelta(days=int(rng.integers(60, 200)))
                frows.append((tk, q, restated, revenue * rng.normal(1, 0.03), eps * rng.normal(1, 0.15)))
    fundamentals = pd.DataFrame(frows, columns=["ticker", "period_end", "filed_date", "revenue", "eps"])

    return PointInTimeStore(prices=prices, universe=universe, fundamentals=fundamentals)
