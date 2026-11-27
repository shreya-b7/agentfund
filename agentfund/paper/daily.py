"""The daily paper-trading routine: what a small systematic fund does after every close.

``run_daily`` is safe to run at any time, any number of times:

* **During market hours it does nothing.** Prices for today aren't final yet.
* **Same prices as last run (weekend, holiday, second run today): nothing to do.**
* **Pending orders** from the last run fill at the next open once that day's prices exist.
* **Rebalances once a month** (first run with a new month's prices), or when the
  account has no positions yet. Other days it only marks the book to market.
* **Kill switch, checked every day:** if the account is down more than the risk
  limit from its peak, the risk officer flattens it, whatever the calendar says.
* **Paused** (``agentfund pause``): fills and marks as usual, but places no new orders.

Every run writes one record to the hash-chained decision log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ..risk import RiskOfficer
from .paper_trader import PaperTrader

NY = ZoneInfo("America/New_York")
MARKET_OPEN, PRICES_FINAL = time(9, 30), time(16, 30)
PAPER_DIR = Path("runs/paper")
CONFIG_PATH = PAPER_DIR / "config.json"
PAUSE_PATH = PAPER_DIR / "PAUSED"
DEFAULT_STRATEGY = "risk_parity"


def market_is_open(now: datetime | None = None) -> bool:
    """True between the open and the time closing prices are final, on weekdays (New York time).

    Holidays are not modelled: on a holiday this returns True during the day, which only means
    the routine waits until the evening. That's the safe direction.
    """
    now = (now or datetime.now(NY)).astimezone(NY)
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < PRICES_FINAL


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {"strategy": DEFAULT_STRATEGY}


def save_config(**updates) -> dict:
    cfg = {**load_config(), **updates}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return cfg


def is_paused() -> bool:
    return PAUSE_PATH.exists()


@dataclass
class DailyReport:
    action: str  # "waited", "no_new_prices", "held", "rebalanced", "kill_switch", "paused"
    prices_to: str
    equity: float
    message: str
    orders: list = field(default_factory=list)
    fills_today: list = field(default_factory=list)


def run_daily(trader: PaperTrader, submit: bool = True, now: datetime | None = None) -> DailyReport:
    prices_to = str(trader.store.dates[-1].date())
    if market_is_open(now):
        return DailyReport("waited", prices_to, float("nan"),
                           "The market is open, so today's prices aren't final. Run again after 4:30pm New York time.")

    fills_before = len(getattr(trader.client, "state", {}).get("fills", []))
    acct = trader.client.connect()  # local: fills pending orders at the next open; Alpaca: verifies paper account
    equity = float(acct["equity"])
    fills = getattr(trader.client, "state", {}).get("fills", [])[fills_before:]
    positions = trader.client.positions()
    state = trader.load_state()
    month = prices_to[:7]

    def finish(action: str, message: str, orders=None) -> DailyReport:
        if submit:  # a dry run must not stop the real run from processing these prices
            trader.save_state(last_processed=prices_to)
        trader.log.append(
            "paper_daily",
            {"action": action, "prices_to": prices_to, "equity": equity, "positions": len(positions),
             "fills": len(fills), "message": message},
            decision_time=pd.Timestamp.now(tz="UTC"),
        )
        return DailyReport(action, prices_to, equity, message, orders or [], fills)

    if submit and state.get("last_processed") == prices_to and not fills:
        return DailyReport("no_new_prices", prices_to, equity,
                           f"Already processed prices up to {prices_to} (weekend, holiday, or a second run today).")
    if trader.client.open_orders():
        return finish("held", "Orders from the last run are still waiting for the next open.")

    peak = trader._peak(equity)
    drawdown = 1 - equity / peak
    limit = trader.risk.limits.drawdown_kill_switch
    if positions and drawdown > limit:
        current = {p["symbol"]: float(p["market_value"]) / equity for p in positions}
        res = trader.run_cycle(submit=submit, proposal=current, connect=False)
        trader.save_state(peak_equity=equity)  # re-arm from the flattened level, as in the backtest
        return finish("kill_switch",
                      f"Down {drawdown:.1%} from the peak (limit {limit:.0%}): the risk officer flattened the account.",
                      res.orders)

    due = not positions or state.get("last_rebalance_month") != month
    if not due:
        return finish("held", f"Holding. Next rebalance with the first prices of next month (last: {state.get('last_rebalance_month')}).")
    if is_paused():
        return finish("paused", "A rebalance is due, but trading is paused. Run 'agentfund resume' to restart.")

    res = trader.run_cycle(submit=submit, connect=False)
    if submit and not res.skipped_reason:
        trader.save_state(last_rebalance_month=month)
    verdict = res.decision.verdict.value
    return finish("rebalanced",
                  f"Monthly rebalance: risk officer {verdict}, {len(res.orders)} orders "
                  f"{'placed for the next open' if submit else '(dry run, nothing placed)'}.",
                  res.orders)


def format_report(r: DailyReport, strategy: str) -> str:
    lines = [f"agentfund daily · strategy {strategy} · prices to {r.prices_to}" + (" · PAUSED" if is_paused() else "")]
    if r.equity == r.equity:  # not NaN
        lines.append(f"paper account value ${r.equity:,.2f} ({(r.equity / 100_000 - 1):+.2%} since start)")
    for f in r.fills_today:
        if f.get("status") == "rejected":
            lines.append(f"  rejected {f['side']} {f['symbol']}: {f.get('reason')}")
        else:
            lines.append(f"  filled {f['side']:4s} {f['symbol']:6s} {abs(f['qty_filled']):.4f} sh @ ${f['price']:.2f} on {f['filled_on']}")
    lines.append(r.message)
    for o in r.orders:
        size = f"${o.notional:,.2f}" if o.notional is not None else f"{o.qty} sh"
        lines.append(f"  {o.side.upper():4s} {o.symbol:6s} {size}")
    return "\n".join(lines)
