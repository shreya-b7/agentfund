"""yfinance conversion and the local (file-based) paper account."""

import numpy as np
import pandas as pd
import pytest

from agentfund.audit import verify
from agentfund.audit import DecisionLog
from agentfund.data import PointInTimeStore
from agentfund.data.yahoo import to_long, yahoo_store
from agentfund.paper import LocalPaperBroker, PaperTrader
from agentfund.strategies import EqualWeight


def fake_yf(symbols, start, end):
    """Same shape yfinance returns: columns MultiIndex (Price, Ticker), index named Date."""
    idx = pd.bdate_range("2023-01-02", periods=300, name="Date")
    rng = np.random.default_rng(0)
    cols, data = [], []
    for field in ("Close", "High", "Low", "Open", "Volume"):
        for k, s in enumerate(symbols):
            cols.append((field, s))
            base = 100 + k * 10 + np.cumsum(rng.normal(0, 1, len(idx)))
            data.append(rng.uniform(1e6, 5e6, len(idx)) if field == "Volume" else base)
    df = pd.DataFrame(np.array(data).T, index=idx, columns=pd.MultiIndex.from_tuples(cols, names=["Price", "Ticker"]))
    return df


SYMS = {"AAPL": ("Apple", "Tech"), "MSFT": ("Microsoft", "Tech"), "JPM": ("JPMorgan", "Fin"), "XOM": ("Exxon", "Energy")}


def test_yfinance_frame_converts_to_store():
    store = yahoo_store(SYMS, downloader=fake_yf)
    assert set(store.prices.ticker) == set(SYMS)
    assert len(store.dates) == 300
    assert {"open", "close", "volume"} <= set(store.prices.columns)


def test_single_ticker_frame():
    raw = fake_yf(["AAPL"], None, None).droplevel("Ticker", axis=1)
    long = to_long(raw, ["AAPL"])
    assert set(long.ticker) == {"AAPL"} and len(long) == 300


def test_empty_download_is_an_error():
    with pytest.raises(RuntimeError):
        to_long(pd.DataFrame(), ["AAPL"])


def _cut(store: PointInTimeStore, end) -> PointInTimeStore:
    return PointInTimeStore(store.prices[store.prices.date <= end], store.universe, store.fundamentals)


def test_local_paper_orders_fill_at_next_open(tmp_path):
    full = yahoo_store(SYMS, downloader=fake_yf)
    day = full.dates[-5]
    today = _cut(full, day)
    acct_file, log = tmp_path / "acct.json", DecisionLog(tmp_path / "d.jsonl")

    broker = LocalPaperBroker(today, acct_file)
    assert broker.connect()["equity"] == 100_000
    trader = PaperTrader(broker, today, EqualWeight(), log=log, state_path=tmp_path / "s.json")
    res = trader.run_cycle(submit=True)
    assert res.submitted and len(broker.open_orders()) == len(res.orders)
    assert broker.positions() == []  # nothing fills until the next day's prices exist

    # Next day's prices arrive (user runs fetch again): orders fill at that open.
    tomorrow = _cut(full, full.dates[-4])
    broker2 = LocalPaperBroker(tomorrow, acct_file)
    acct = broker2.connect()
    assert broker2.open_orders() == []
    fills = broker2.state["fills"]
    nxt = full.dates[-4]
    opens = full.wide("open")
    assert all(f["filled_on"] == str(nxt.date()) and f["price"] == pytest.approx(opens.loc[nxt, f["symbol"]]) for f in fills)
    assert acct["cash"] < 100_000 and len(broker2.positions()) == len(fills)
    # Risk limit still applies: no position above ~10% of equity at fill.
    assert max(p["market_value"] for p in broker2.positions()) < 0.12 * acct["equity"]
    assert verify(log.path)[0]


def test_local_account_never_goes_through_a_network():
    import agentfund.paper.local_broker as lb
    src = open(lb.__file__).read()
    assert "httpx" not in src and "requests" not in src and "http" not in src.replace("https://", "")


def test_local_paper_cash_never_goes_negative(tmp_path):
    from agentfund.broker.alpaca import Order

    full = yahoo_store(SYMS, downloader=fake_yf)
    today = _cut(full, full.dates[-5])
    broker = LocalPaperBroker(today, tmp_path / "a.json", starting_cash=10_000)
    broker.connect()
    for s in SYMS:  # ask for far more than the account holds
        broker.submit(Order(s, "buy", notional=9_000))
    after = LocalPaperBroker(_cut(full, full.dates[-4]), tmp_path / "a.json")
    acct = after.connect()
    assert acct["cash"] >= 0
    assert any(f.get("status") == "rejected" for f in after.state["fills"])
