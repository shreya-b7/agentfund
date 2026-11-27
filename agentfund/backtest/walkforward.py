"""Walk-forward testing: choose a variant on past data only, then judge it on the next unseen period.

Each variant is backtested once over the whole sample. Then, window by window,
the variant with the best Sharpe over the trailing ``train_days`` is chosen and
its returns over the following ``test_days`` are kept. Stitching those test
windows together gives an out-of-sample return series that never benefited from
knowing which variant would turn out best.

Simplification: switching between variants is treated as free (each variant's
own trading costs are included, but not the cost of moving from one variant's
book to another's).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..data.pit import PointInTimeStore
from .engine import Backtester
from .metrics import deflated_sharpe, max_drawdown, sharpe


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    choices: pd.DataFrame  # one row per test window
    in_sample_best: str  # what you'd pick looking at the whole sample (with hindsight)
    variant_sharpes: dict[str, float]

    def summary(self) -> dict[str, float]:
        eq = (1 + self.oos_returns).cumprod()
        return {
            "oos_sharpe": sharpe(self.oos_returns),
            "oos_max_drawdown": max_drawdown(eq),
            "oos_deflated_sharpe": deflated_sharpe(self.oos_returns, len(self.variant_sharpes)),
            "hindsight_best_sharpe": self.variant_sharpes[self.in_sample_best],
            "windows": len(self.choices),
        }


def walk_forward(
    store: PointInTimeStore,
    variants: dict[str, Callable[[], object]],
    train_days: int = 756,
    test_days: int = 126,
    start=None,
    end=None,
) -> WalkForwardResult:
    rets = {name: Backtester(store, make()).run(start, end).returns for name, make in variants.items()}
    df = pd.DataFrame(rets).fillna(0.0)
    dates = df.index
    pieces, rows = [], []
    for i in range(train_days, len(dates), test_days):
        train = df.iloc[i - train_days : i]
        test = df.iloc[i : i + test_days]
        scores = {c: sharpe(train[c]) for c in df.columns}
        pick = max(scores, key=scores.get)
        pieces.append(test[pick])
        rows.append(
            {
                "test_start": test.index[0].date(),
                "test_end": test.index[-1].date(),
                "chosen": pick,
                "train_sharpe": scores[pick],
                "test_sharpe": sharpe(test[pick]),
            }
        )
    if not pieces:
        raise ValueError("sample too short for one train + test window")
    full = {c: sharpe(df[c]) for c in df.columns}
    return WalkForwardResult(
        oos_returns=pd.concat(pieces),
        choices=pd.DataFrame(rows),
        in_sample_best=max(full, key=full.get),
        variant_sharpes=full,
    )


def momentum_grid() -> dict[str, Callable[[], object]]:
    from ..strategies import Momentum

    return {
        f"mom_lb{lb}_top{int(tf * 100)}": (lambda lb=lb, tf=tf: Momentum(lookback=lb, top_frac=tf, min_history=lb))
        for lb in (126, 252)
        for tf in (0.1, 0.2, 0.3)
    }


def strategy_grid() -> dict[str, Callable[[], object]]:
    from ..strategies import STRATEGIES

    return {name: cls for name, cls in STRATEGIES.items()}
