"""Event-driven daily backtester.

Timeline for each trading day ``t``:

1. Names that stopped trading are liquidated at their last close (minus a
   delisting haircut).
2. Orders decided at the close of ``t-1`` fill at the open of ``t``, paying
   commission, half the spread and square-root market impact.
3. The book is marked to market at the close of ``t``.
4. The strategy sees a :class:`~agentfund.data.pit.PITView` as of ``t`` and may
   propose new target weights. The risk officer reviews them and every review is
   written to the decision log. Approved targets fill at the next open.

A decision at ``t`` can therefore never trade at a price it was computed from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from ..audit import DecisionLog
from ..data.pit import PITView, PointInTimeStore
from ..risk import RiskContext, RiskOfficer


class Strategy(Protocol):
    name: str

    def on_bar(self, view: PITView) -> dict[str, float] | None:
        """Return target weights to rebalance to, or ``None`` to hold."""


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    impact_coef: float = 0.1  # cost fraction = coef * sqrt(notional / ADV$)
    delisting_haircut: float = 0.30  # fraction lost when a held name delists

    def cost(self, notional: float, adv_dollars: float) -> float:
        n = abs(notional)
        if n == 0:
            return 0.0
        linear = n * (self.commission_bps + self.half_spread_bps) / 1e4
        participation = n / adv_dollars if adv_dollars > 0 else 1.0
        return linear + n * self.impact_coef * math.sqrt(participation)


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    decisions: list[dict] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())


def adv_dollars(view: PITView, window: int = 20) -> dict[str, float]:
    px = view.history("close", lookback=window)
    vol = view.history("volume", lookback=window)
    adv = (px * vol).mean(skipna=True)
    return {k: float(v) for k, v in adv.items() if pd.notna(v)}


class Backtester:
    def __init__(
        self,
        store: PointInTimeStore,
        strategy: Strategy,
        risk: RiskOfficer | None = None,
        costs: CostModel | None = None,
        initial_capital: float = 1_000_000.0,
        log: DecisionLog | None = None,
    ):
        self.store = store
        self.strategy = strategy
        self.risk = risk or RiskOfficer()
        self.costs = costs or CostModel()
        self.capital = initial_capital
        self.log = log

    def run(self, start=None, end=None) -> BacktestResult:
        dates = self.store.dates
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        opens, closes = self.store.wide("open"), self.store.wide("close")

        cash = self.capital
        shares: dict[str, float] = {}
        last_close: dict[str, float] = {}
        pending: dict[str, float] | None = None
        pending_adv: dict[str, float] = {}
        peak = self.capital

        eq_hist, to_hist, cost_hist, trades, decisions = [], [], [], [], []

        for t in dates:
            o_row, c_row = opens.loc[t], closes.loc[t]
            day_cost, day_traded = 0.0, 0.0

            # 1. Delistings: held names with no price today.
            for tk in [k for k in shares if pd.isna(c_row.get(k, np.nan))]:
                proceeds = shares[tk] * last_close[tk] * (1 - self.costs.delisting_haircut)
                trades.append((t, tk, -shares[tk], last_close[tk], "delisted", 0.0))
                cash += proceeds
                del shares[tk]

            # 2. Fill yesterday's decision at today's open.
            if pending is not None:
                equity_open = cash + sum(n * o_row[k] for k, n in shares.items())
                for tk in sorted(set(pending) | set(shares)):
                    px = o_row.get(tk, np.nan)
                    if pd.isna(px) or px <= 0:
                        continue
                    target = pending.get(tk, 0.0) * equity_open / px
                    d = target - shares.get(tk, 0.0)
                    if abs(d * px) < 1.0:
                        continue
                    notional = d * px
                    c = self.costs.cost(notional, pending_adv.get(tk, 0.0))
                    cash -= notional + c
                    day_cost += c
                    day_traded += abs(notional)
                    shares[tk] = target
                    trades.append((t, tk, d, px, "rebalance", c))
                shares = {k: v for k, v in shares.items() if abs(v) > 1e-9}
                pending = None

            # 3. Mark to market.
            for k in shares:
                last_close[k] = c_row[k]
            equity = cash + sum(n * c_row[k] for k, n in shares.items())
            peak = max(peak, equity)
            eq_hist.append(equity)
            to_hist.append(day_traded / equity if equity > 0 else 0.0)
            cost_hist.append(day_cost)

            # 4. Decide at the close.
            view = self.store.view(t)
            proposal = self.strategy.on_bar(view)
            cur_w = {k: n * c_row[k] / equity for k, n in shares.items()}
            if proposal is None:
                # The kill switch is monitored daily, not only on rebalance days.
                if not (shares and 1 - equity / peak > self.risk.limits.drawdown_kill_switch):
                    continue
                proposal = dict(cur_w)
            adv = adv_dollars(view)
            ctx = RiskContext(
                current_weights=cur_w,
                prices={k: float(v) for k, v in c_row.dropna().items()},
                adv_dollars=adv,
                sectors=view.sectors(),
                tradeable=set(view.tickers()),
                equity=equity,
                drawdown=1 - equity / peak,
            )
            decision = self.risk.review(proposal, ctx)
            rec = {"date": str(t.date()), "strategy": self.strategy.name, **decision.to_dict()}
            decisions.append(rec)
            if self.log is not None:
                self.log.append("risk_review", rec, decision_time=t)
            pending, pending_adv = decision.approved, adv
            if any(b.rule == "drawdown_kill_switch" for b in decision.breaches):
                peak = equity  # re-arm: drawdown is measured from the flatten point

        idx = pd.DatetimeIndex(dates)
        equity_s = pd.Series(eq_hist, index=idx, name="equity")
        trades_df = pd.DataFrame(trades, columns=["date", "ticker", "shares", "price", "reason", "cost"])
        return BacktestResult(
            strategy=self.strategy.name,
            equity=equity_s,
            returns=equity_s.pct_change().fillna(0.0),
            trades=trades_df,
            turnover=pd.Series(to_hist, index=idx, name="turnover"),
            costs=pd.Series(cost_hist, index=idx, name="costs"),
            decisions=decisions,
        )
