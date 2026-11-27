"""A simulated paper-trading account that lives in a JSON file on your computer.

No broker, no account, no keys, no bank: it cannot touch real money because it
never talks to anything that holds money. It starts with $100,000 of play money.

How fills work (same rule as the backtester): orders you submit wait as
"pending" and fill at the **open of the next trading day** after they were
placed, at real prices from the price store, paying the same commission, spread
and market-impact costs as the backtest. Positions are valued at the latest close.
Run ``fetch`` each day so the store has the newest prices.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..backtest.engine import CostModel
from ..broker.alpaca import Order
from ..data.pit import PointInTimeStore

ACCOUNT_NUMBER = "LOCAL-PAPER"


class LocalPaperBroker:
    def __init__(
        self,
        store: PointInTimeStore,
        path: str | Path = "runs/paper/local_account.json",
        starting_cash: float = 100_000.0,
        costs: CostModel | None = None,
    ):
        self.store, self.path, self.costs = store, Path(path), costs or CostModel()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": starting_cash, "positions": {}, "pending": [], "fills": [], "equity_history": []}
            self._save()

    # ----- helpers ----------------------------------------------------
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, default=str))

    def _close(self) -> pd.Series:
        return self.store.wide("close").ffill().iloc[-1]

    def _fill_pending(self) -> None:
        """Fill every pending order whose next trading day's open is now in the price store."""
        opens = self.store.wide("open")
        closes, vols = self.store.wide("close"), self.store.wide("volume")
        still = []
        # Sells first so their proceeds can fund the buys, like a real rebalance.
        ordered = sorted(self.state["pending"], key=lambda o: o["side"] != "sell")
        for o in ordered:
            placed = pd.Timestamp(o["placed_on"])
            later = opens.index[opens.index > placed]
            px = opens.loc[later[0], o["symbol"]] if len(later) else float("nan")
            if not len(later) or pd.isna(px):
                still.append(o)
                continue
            day = later[0]
            adv = float((closes[o["symbol"]] * vols[o["symbol"]]).loc[:placed].tail(20).mean())
            pos = self.state["positions"].get(o["symbol"], {"qty": 0.0, "cost_basis": 0.0})
            if o["side"] == "buy":
                # Never spend more cash than the account has (prices moved overnight, plus costs).
                budget = min(o["notional"], max(self.state["cash"], 0.0) / (1 + 0.002))
                if budget < 1.0:
                    self.state["fills"].append({**o, "status": "rejected", "reason": "insufficient cash"})
                    continue
                qty = budget / px
            else:
                qty = -min(o["qty"], pos["qty"])
            notional = qty * px
            cost = self.costs.cost(notional, adv)
            self.state["cash"] -= notional + cost
            if qty > 0:
                pos["cost_basis"] += notional
            elif pos["qty"] > 0:
                pos["cost_basis"] *= 1 + qty / pos["qty"]
            pos["qty"] += qty
            if pos["qty"] <= 1e-9:
                self.state["positions"].pop(o["symbol"], None)
            else:
                self.state["positions"][o["symbol"]] = pos
            self.state["fills"].append(
                {**o, "filled_on": str(day.date()), "price": float(px), "qty_filled": qty, "cost": cost}
            )
        self.state["pending"] = still

    # ----- broker interface (same shape as AlpacaPaperClient) ---------
    def connect(self) -> dict:
        self._fill_pending()
        acct = self.account()
        hist = self.state["equity_history"]
        last_date = str(self.store.dates[-1].date())
        if not hist or hist[-1]["date"] != last_date:
            hist.append({"date": last_date, "equity": acct["equity"]})
        self._save()
        return acct

    def account(self) -> dict:
        equity = self.state["cash"] + sum(p["market_value"] for p in self.positions())
        return {"account_number": ACCOUNT_NUMBER, "equity": equity, "cash": self.state["cash"], "status": "SIMULATED"}

    def positions(self) -> list[dict]:
        close = self._close()
        out = []
        for sym, p in sorted(self.state["positions"].items()):
            px = float(close.get(sym, float("nan")))
            mv = p["qty"] * px
            out.append(
                {
                    "symbol": sym,
                    "qty": p["qty"],
                    "current_price": px,
                    "market_value": mv,
                    "unrealized_pl": mv - p["cost_basis"],
                }
            )
        return out

    def open_orders(self) -> list[dict]:
        return list(self.state["pending"])

    def clock(self) -> dict:
        return {"is_open": False, "next_open": "fills at the next trading day's open after you run fetch"}

    def submit(self, order: Order) -> dict:
        rec = {
            "id": f"local-{uuid.uuid4().hex[:12]}",
            "symbol": order.symbol,
            "side": order.side,
            "notional": order.notional,
            "qty": order.qty,
            "placed_on": str(self.store.dates[-1].date()),
            "placed_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        self.state["pending"].append(rec)
        self._save()
        return rec
