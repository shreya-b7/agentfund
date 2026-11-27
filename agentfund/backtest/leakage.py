"""Look-ahead leakage tests.

The core check is *future-perturbation invariance*: scramble every piece of data
that became knowable after a cutoff date, rerun, and require that every decision
made on or before the cutoff is bit-for-bit identical. A strategy, agent or
engine change that peeks at the future (directly, through a cached global, a
centred rolling window, a full-sample normalisation, ...) fails this test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..data.pit import PointInTimeStore
from .engine import Backtester, Strategy


def perturb_future(store: PointInTimeStore, cutoff, seed: int = 0) -> PointInTimeStore:
    """Copy of ``store`` with every price and filing after ``cutoff`` randomly changed."""
    cutoff = pd.Timestamp(cutoff)
    rng = np.random.default_rng(seed)

    p = store.prices.copy()
    future = p["date"] > cutoff
    shock = np.exp(rng.normal(0, 0.2, future.sum()))
    for col in ("open", "high", "low", "close"):
        p.loc[future, col] = p.loc[future, col].to_numpy() * shock
    p.loc[future, "volume"] = p.loc[future, "volume"].to_numpy() * rng.uniform(0.2, 5.0, future.sum())

    f = store.fundamentals.copy()
    ffut = f["filed_date"] > cutoff
    for col in [c for c in f.columns if c not in ("ticker", "period_end", "filed_date")]:
        if pd.api.types.is_numeric_dtype(f[col]):
            f.loc[ffut, col] = f.loc[ffut, col].to_numpy() * rng.normal(1, 0.5, ffut.sum())

    return PointInTimeStore(prices=p, universe=store.universe.copy(), fundamentals=f)


@dataclass
class LeakageReport:
    passed: bool
    checked_dates: int
    first_divergence: pd.Timestamp | None
    detail: str


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) <= 1e-12 for k in a)


def check_strategy(
    strategy_factory: Callable[[PointInTimeStore], Strategy],
    store: PointInTimeStore,
    cutoff,
    start=None,
    seed: int = 0,
) -> LeakageReport:
    """Run a strategy's decisions on real vs future-perturbed data up to ``cutoff``."""
    cutoff = pd.Timestamp(cutoff)
    other = perturb_future(store, cutoff, seed)
    s1, s2 = strategy_factory(store), strategy_factory(other)
    dates = store.dates[(store.dates <= cutoff) & ((store.dates >= pd.Timestamp(start)) if start else True)]
    for t in dates:
        a, b = s1.on_bar(store.view(t)), s2.on_bar(other.view(t))
        if not _same(a, b):
            return LeakageReport(False, len(dates), t, f"decision on {t.date()} depends on data after {cutoff.date()}")
    return LeakageReport(True, len(dates), None, "decisions are invariant to future data")


def check_backtest(
    strategy_factory: Callable[[PointInTimeStore], Strategy],
    store: PointInTimeStore,
    cutoff,
    start=None,
    seed: int = 0,
) -> LeakageReport:
    """End-to-end: the equity curve up to ``cutoff`` must not change when the future does."""
    cutoff = pd.Timestamp(cutoff)
    other = perturb_future(store, cutoff, seed)
    r1 = Backtester(store, strategy_factory(store)).run(start=start)
    r2 = Backtester(other, strategy_factory(other)).run(start=start)
    e1, e2 = r1.equity.loc[:cutoff], r2.equity.loc[:cutoff]
    diff = (e1 - e2).abs() > 1e-6 * e1.abs()
    if diff.any():
        t = diff.idxmax()
        return LeakageReport(False, len(e1), t, f"equity on {t.date()} depends on data after {cutoff.date()}")
    return LeakageReport(True, len(e1), None, "equity curve is invariant to future data")
