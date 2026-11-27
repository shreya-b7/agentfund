"""Performance metrics, including the probabilistic and deflated Sharpe ratios.

PSR and DSR follow Bailey & Lopez de Prado (2012, 2014). DSR asks: given that
``n_trials`` variants were tried, how likely is it that this Sharpe ratio beats
the best one we would expect from luck alone?
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def sortino(returns: pd.Series, periods: int = 252) -> float:
    r = returns.dropna()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / downside * math.sqrt(periods)) if downside > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((1 - equity / equity.cummax()).max())


def cagr(equity: pd.Series, periods: int = 252) -> float:
    years = (len(equity) - 1) / periods
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``), adjusting for skew and kurtosis."""
    r = returns.dropna()
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or sd == 0:
        return 0.0
    sr = r.mean() / sd
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas gives excess kurtosis
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return 0.0
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    a = (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
    b = EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * (a + b)


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None) -> float:
    """DSR. ``trial_sharpes`` are per-period Sharpes of all variants tried (for their variance).

    If not given, the variance of the Sharpe estimator itself, 1/(T-1), is used,
    which is a conservative stand-in for independent trials.
    """
    r = returns.dropna()
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var = float(np.var(trial_sharpes, ddof=1))
    else:
        var = 1.0 / max(len(r) - 1, 1)
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, var))


def summarize(result, n_trials: int = 1) -> dict[str, float]:
    eq, r = result.equity, result.returns
    years = max((len(eq) - 1) / 252, 1e-9)
    return {
        "cagr": cagr(eq),
        "vol": float(r.std(ddof=1) * math.sqrt(252)),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(eq),
        "annual_turnover": float(result.turnover.sum() / years),
        "cost_drag_bps_per_year": float(result.costs.sum() / eq.mean() / years * 1e4),
        "psr": probabilistic_sharpe(r),
        "deflated_sharpe": deflated_sharpe(r, n_trials),
        "n_trades": int(len(result.trades)),
    }
