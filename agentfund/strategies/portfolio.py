"""Portfolio construction: risk parity and minimum variance on a shrunk covariance matrix.

Sample covariance matrices from a year of daily data are noisy, and optimizers
latch onto the noise. Ledoit-Wolf shrinkage pulls the estimate toward a
structured target (here: a scaled identity), with the shrinkage amount chosen
from the data. Both optimizers are long-only and respect a per-name cap, so the
risk officer does not have to resize their output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.pit import PITView
from .baselines import _Monthly


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004) shrinkage toward ``mu * I``. Returns (covariance, shrinkage in [0, 1])."""
    x = returns - returns.mean(axis=0)
    t, n = x.shape
    s = x.T @ x / t
    mu = np.trace(s) / n
    target = mu * np.eye(n)
    d2 = np.sum((s - target) ** 2)
    # sum_t ||x_t x_t' - S||^2 = sum_t ||x_t||^4 - t ||S||^2  (avoids T outer products)
    b2 = (np.sum(np.sum(x**2, axis=1) ** 2) - t * np.sum(s**2)) / t**2
    shrink = float(min(b2, d2) / d2) if d2 > 0 else 1.0
    return shrink * target + (1 - shrink) * s, shrink


def project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {w : sum w = 1, 0 <= w <= cap}."""
    lo, hi = v.min() - cap, v.max()
    for _ in range(50):
        tau = (lo + hi) / 2
        if np.clip(v - tau, 0, cap).sum() > 1:
            lo = tau
        else:
            hi = tau
    return np.clip(v - (lo + hi) / 2, 0, cap)


def min_variance(cov: np.ndarray, cap: float = 0.10, iters: int = 500) -> np.ndarray:
    """Long-only minimum variance with a per-name cap (projected gradient descent)."""
    n = len(cov)
    cap = max(cap, 1.0 / n)
    w = np.full(n, 1.0 / n)
    step = 1.0 / np.linalg.eigvalsh(cov).max()
    for _ in range(iters):
        w_new = project_capped_simplex(w - step * 2 * cov @ w, cap)
        if np.abs(w_new - w).max() < 1e-8:
            break
        w = w_new
    return w


def risk_parity(cov: np.ndarray, iters: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Equal risk contribution weights (each name contributes the same share of portfolio variance).

    Cyclical coordinate descent on  min 1/2 y'Cy - (1/n) sum(log y),  whose solution, normalized,
    is the ERC portfolio (Griveau-Billion, Richard & Roncalli, 2013). Each coordinate update is the
    positive root of a quadratic, so weights stay strictly positive even with negative correlations.
    """
    n = len(cov)
    diag = np.diag(cov)
    y = 1 / np.sqrt(diag)
    for _ in range(iters):
        prev = y.copy()
        for i in range(n):
            c = cov[i] @ y - diag[i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * diag[i] / n)) / (2 * diag[i])
        if np.abs(y - prev).max() < tol * np.abs(y).max():
            break
    return y / y.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    rc = w * (cov @ w)
    return rc / rc.sum()


class _CovStrategy(_Monthly):
    def __init__(self, window: int = 252, max_names: int = 30, cap: float = 0.10):
        super().__init__()
        self.window, self.max_names, self.cap = window, max_names, cap
        self.last_shrinkage: float | None = None

    def _inputs(self, view: PITView) -> tuple[list[str], np.ndarray] | None:
        px = view.history("close", lookback=self.window + 1)
        if len(px) < self.window + 1:
            return None
        px = px[[t for t in view.tickers() if t in px.columns]].dropna(axis=1)
        dv = (view.history("close", 20) * view.history("volume", 20)).mean()
        names = list(dv.reindex(px.columns).dropna().sort_values(ascending=False).index[: self.max_names])
        if len(names) < 2:
            return None
        rets = px[names].pct_change().dropna().to_numpy()
        cov, self.last_shrinkage = ledoit_wolf(rets)
        return names, cov

    def weights(self, view: PITView) -> dict[str, float]:
        got = self._inputs(view)
        if got is None:
            return {}
        names, cov = got
        w = self.optimize(cov)
        return {n: float(x) for n, x in zip(names, w) if x > 1e-6}

    def optimize(self, cov: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class MinVariance(_CovStrategy):
    name = "min_variance"

    def optimize(self, cov):
        return min_variance(cov, self.cap)


class RiskParity(_CovStrategy):
    name = "risk_parity"

    def optimize(self, cov):
        w = risk_parity(cov)
        if w.max() > self.cap:  # cap, then hand the excess to the uncapped names pro rata
            w = project_capped_simplex(w, self.cap)
        return w
