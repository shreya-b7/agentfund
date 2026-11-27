"""Baseline strategies. Each rebalances on the first trading day of a new month."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.pit import PITView


class _Monthly:
    name = "base"

    def __init__(self) -> None:
        self._last_month: tuple[int, int] | None = None

    def _due(self, view: PITView) -> bool:
        ym = (view.as_of.year, view.as_of.month)
        if ym == self._last_month:
            return False
        self._last_month = ym
        return True

    def on_bar(self, view: PITView) -> dict[str, float] | None:
        if not self._due(view):
            return None
        return self.weights(view)

    def weights(self, view: PITView) -> dict[str, float]:  # pragma: no cover - abstract
        raise NotImplementedError


class EqualWeight(_Monthly):
    """Equal weight across the current universe, up to ``max_names`` by liquidity."""

    name = "equal_weight"

    def __init__(self, max_names: int = 30):
        super().__init__()
        self.max_names = max_names

    def weights(self, view: PITView) -> dict[str, float]:
        names = view.tickers()
        dv = (view.history("close", 20) * view.history("volume", 20)).mean()
        names = [n for n in dv.reindex(names).dropna().sort_values(ascending=False).index[: self.max_names]]
        return {n: 1.0 / len(names) for n in names} if names else {}


class Momentum(_Monthly):
    """12-1 month cross-sectional momentum: equal weight the top ``top_frac`` of names."""

    name = "momentum_12_1"

    def __init__(self, lookback: int = 252, skip: int = 21, top_frac: float = 0.2, min_history: int = 252):
        super().__init__()
        self.lookback, self.skip, self.top_frac, self.min_history = lookback, skip, top_frac, min_history

    def weights(self, view: PITView) -> dict[str, float]:
        px = view.history("close", lookback=self.lookback + 1)
        if len(px) < self.lookback + 1:
            return {}
        px = px[[c for c in view.tickers() if c in px.columns]]
        full = px.notna().sum() >= self.min_history
        mom = (px.iloc[-1 - self.skip] / px.iloc[0] - 1)[full].dropna()
        if mom.empty:
            return {}
        k = max(1, int(len(mom) * self.top_frac))
        winners = mom.sort_values(ascending=False).index[:k]
        return {t: 1.0 / k for t in winners}


class InverseVol(_Monthly):
    """Inverse 60-day volatility weights across the most liquid names."""

    name = "inverse_vol"

    def __init__(self, window: int = 60, max_names: int = 30):
        super().__init__()
        self.window, self.max_names = window, max_names

    def weights(self, view: PITView) -> dict[str, float]:
        px = view.history("close", lookback=self.window + 1)[view.tickers()]
        vol = px.pct_change(fill_method=None).std().replace(0, np.nan).dropna()
        dv = (view.history("close", 20) * view.history("volume", 20)).mean()
        names = dv.reindex(vol.index).dropna().sort_values(ascending=False).index[: self.max_names]
        inv = 1 / vol[names]
        w = inv / inv.sum()
        return {k: float(v) for k, v in w.items() if pd.notna(v)}


def _all():
    from .portfolio import MinVariance, RiskParity

    return {cls.name: cls for cls in (EqualWeight, Momentum, InverseVol, RiskParity, MinVariance)}


STRATEGIES = _all()
