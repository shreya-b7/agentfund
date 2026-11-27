"""The daily routine: market-hours guard, idempotence, monthly rebalance, pause, kill switch, scheduling."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import agentfund.paper.daily as daily
from agentfund.audit import DecisionLog, verify
from agentfund.data import PointInTimeStore
from agentfund.data.yahoo import drop_unfinished_day
from agentfund.paper import LocalPaperBroker, PaperTrader, run_daily
from agentfund.paper import schedule
from agentfund.strategies import RiskParity

NY = ZoneInfo("America/New_York")
EVENING = datetime(2026, 11, 20, 18, 0, tzinfo=NY)  # a Friday after the close
MIDDAY = datetime(2026, 11, 20, 12, 0, tzinfo=NY)


@pytest.fixture
def world(store, tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "PAUSE_PATH", tmp_path / "PAUSED")
    full = store

    def trader_on(day):
        cut = PointInTimeStore(full.prices[full.prices.date <= day], full.universe, full.fundamentals)
        broker = LocalPaperBroker(cut, tmp_path / "acct.json")
        return PaperTrader(broker, cut, RiskParity(), log=DecisionLog(tmp_path / "log.jsonl"),
                           state_path=tmp_path / "state.json")

    return full, trader_on, tmp_path


def _days(full, start, n):
    d = full.dates
    i = d.searchsorted(pd.Timestamp(start))
    return list(d[i : i + n])


def test_month_of_daily_runs(world):
    full, trader_on, tmp = world
    days = _days(full, "2019-01-28", 30)  # crosses into February and March

    r = run_daily(trader_on(days[0]), now=EVENING)
    assert r.action == "rebalanced" and r.orders  # empty account: invest right away
    assert run_daily(trader_on(days[0]), now=EVENING).action == "no_new_prices"  # second run, same prices

    r = run_daily(trader_on(days[1]), now=EVENING)
    assert r.fills_today and r.action == "held"  # filled at the next open, then hold
    actions = [r.action]
    for d in days[2:]:
        rep = run_daily(trader_on(d), now=EVENING)
        actions.append(rep.action)
    assert actions.count("rebalanced") == 2  # first prices of February and of March
    assert verify(tmp / "log.jsonl")[0]


def test_waits_while_market_is_open(world):
    full, trader_on, _ = world
    r = run_daily(trader_on(full.dates[300]), now=MIDDAY)
    assert r.action == "waited"


def test_dry_run_places_nothing_and_does_not_block_the_real_run(world):
    full, trader_on, _ = world
    day = full.dates[300]
    dry = run_daily(trader_on(day), submit=False, now=EVENING)
    assert dry.action == "rebalanced" and trader_on(day).client.open_orders() == []
    real = run_daily(trader_on(day), now=EVENING)
    assert real.action == "rebalanced" and trader_on(day).client.open_orders()


def test_pause_blocks_new_orders(world):
    full, trader_on, tmp = world
    (tmp / "PAUSED").write_text("x")
    r = run_daily(trader_on(full.dates[300]), now=EVENING)
    assert r.action == "paused" and trader_on(full.dates[300]).client.open_orders() == []


def test_kill_switch_flattens_on_any_day(world):
    full, trader_on, tmp = world
    days = _days(full, "2019-01-28", 4)
    run_daily(trader_on(days[0]), now=EVENING)
    run_daily(trader_on(days[1]), now=EVENING)  # fills
    t = trader_on(days[2])
    t.save_state(peak_equity=1_000_000)  # pretend we were 10x higher: deep drawdown
    r = run_daily(t, now=EVENING)
    assert r.action == "kill_switch"
    assert r.orders and all(o.side == "sell" for o in r.orders)


def test_market_hours():
    assert daily.market_is_open(MIDDAY)
    assert not daily.market_is_open(EVENING)
    assert not daily.market_is_open(datetime(2026, 11, 21, 12, 0, tzinfo=NY))  # Saturday


def test_unfinished_day_is_dropped():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-11-19", "2026-11-20"]), "ticker": "A",
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    assert len(drop_unfinished_day(df, MIDDAY)) == 1
    assert len(drop_unfinished_day(df, EVENING)) == 2


def test_schedule_plist_and_protected_folders(monkeypatch, tmp_path):
    p = schedule.build_plist(Path("/Users/me/agentfund"), "/Users/me/agentfund/.venv/bin/python", 17, 0)
    assert p["ProgramArguments"][1:] == ["-m", "agentfund", "daily"]
    assert [d["Weekday"] for d in p["StartCalendarInterval"]] == [1, 2, 3, 4, 5]
    assert p["WorkingDirectory"] == "/Users/me/agentfund"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert schedule.protected_folder(tmp_path / "Downloads" / "agentfund") == "Downloads"
    assert schedule.protected_folder(tmp_path / "agentfund") is None
    h, m = schedule.default_local_time()
    assert 0 <= h < 24 and 0 <= m < 60
