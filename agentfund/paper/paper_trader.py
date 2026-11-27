"""One paper-trading rebalance cycle.

Same decision path as the backtester: strategy -> risk officer -> orders, with
every step written to the hash-chained decision log. Orders go to the Alpaca
**paper** account only. Placed after the close, market orders queue for the next
open, which is exactly how the backtester fills.

Default is a dry run: it prints and logs the orders it *would* send.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..audit import DecisionLog
from ..backtest.engine import adv_dollars
from ..broker.alpaca import AlpacaPaperClient, Order
from ..data.pit import PointInTimeStore
from ..risk import RiskContext, RiskDecision, RiskOfficer


@dataclass
class CycleResult:
    as_of: str
    equity: float
    decision: RiskDecision
    orders: list[Order]
    submitted: list[dict] = field(default_factory=list)
    dry_run: bool = True
    skipped_reason: str | None = None


def plan_orders(
    target: dict[str, float],
    positions: dict[str, dict],
    equity: float,
    prices: dict[str, float],
    min_trade: float = 50.0,
) -> list[Order]:
    """Turn target weights into orders. Sells (by share qty) first, then buys (by dollar amount)."""
    sells, buys = [], []
    for sym in sorted(set(target) | set(positions)):
        held_value = float(positions.get(sym, {}).get("market_value", 0.0))
        held_qty = float(positions.get(sym, {}).get("qty", 0.0))
        delta = target.get(sym, 0.0) * equity - held_value
        if abs(delta) < min_trade and target.get(sym, 0.0) > 0:
            continue
        if delta < 0 and held_qty > 0:
            px = prices.get(sym) or float(positions[sym].get("current_price", 0.0))
            qty = held_qty if target.get(sym, 0.0) == 0 else min(held_qty, -delta / px)
            if qty * px >= min_trade or target.get(sym, 0.0) == 0:
                sells.append(Order(sym, "sell", qty=round(qty, 6)))
        elif delta >= min_trade:
            buys.append(Order(sym, "buy", notional=round(delta, 2)))
    return sells + buys


class PaperTrader:
    def __init__(
        self,
        client: AlpacaPaperClient,
        store: PointInTimeStore,
        strategy,
        risk: RiskOfficer | None = None,
        log: DecisionLog | None = None,
        state_path: str | Path = "runs/paper/state.json",
    ):
        self.client, self.store, self.strategy = client, store, strategy
        self.risk = risk or RiskOfficer()
        self.log = log or DecisionLog("runs/paper/decisions.jsonl")
        self.state_path = Path(state_path)

    def load_state(self) -> dict:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def save_state(self, **updates) -> dict:
        state = {**self.load_state(), **updates}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))
        return state

    def _peak(self, equity: float) -> float:
        peak = max(float(self.load_state().get("peak_equity", equity)), equity)
        self.save_state(peak_equity=peak)
        return peak

    def run_cycle(self, submit: bool = False, proposal: dict[str, float] | None = None, connect: bool = True) -> CycleResult:
        """Strategy (or a given ``proposal``) -> risk officer -> orders, all logged."""
        acct = self.client.connect() if connect else self.client.account()  # connect: verifies paper / fills
        equity = float(acct["equity"])
        positions = {p["symbol"]: p for p in self.client.positions()}
        view = self.store.view(self.store.dates[-1])

        if proposal is None:
            weights_fn = getattr(self.strategy, "weights", None)
            proposal = weights_fn(view) if weights_fn else (self.strategy.on_bar(view) or {})

        prices = {k: float(v) for k, v in view.latest("close").dropna().items()}
        for sym, p in positions.items():
            prices.setdefault(sym, float(p.get("current_price", 0.0)))
        ctx = RiskContext(
            current_weights={s: float(p["market_value"]) / equity for s, p in positions.items()},
            prices=prices,
            adv_dollars=adv_dollars(view),
            sectors=view.sectors(),
            tradeable=set(view.tickers()),
            equity=equity,
            drawdown=1 - equity / self._peak(equity),
        )
        decision = self.risk.review(proposal, ctx)
        orders = plan_orders(decision.approved, positions, equity, prices)
        result = CycleResult(str(view.as_of.date()), equity, decision, orders, dry_run=not submit)

        if submit and self.client.open_orders():
            result.skipped_reason = "open orders already pending; not sending more"
        self.log.append(
            "paper_cycle",
            {
                "as_of": result.as_of,
                "strategy": getattr(self.strategy, "name", "?"),
                "account": acct.get("account_number"),
                "equity": equity,
                "risk": decision.to_dict(),
                "orders": [o.__dict__ for o in orders],
                "submit": submit,
                "skipped_reason": result.skipped_reason,
            },
            decision_time=pd.Timestamp.now(tz="UTC"),
        )
        if submit and not result.skipped_reason:
            for o in orders:
                resp = self.client.submit(o)
                result.submitted.append(resp)
                self.log.append(
                    "paper_order",
                    {"order": o.__dict__, "alpaca_id": resp.get("id"), "status": resp.get("status")},
                    decision_time=pd.Timestamp.now(tz="UTC"),
                )
        return result
