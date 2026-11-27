"""Paper trading: the paper-only lock, order planning, and a full cycle against a mock Alpaca."""

import json

import httpx
import pandas as pd
import pytest

from agentfund.broker import AlpacaPaperClient, NotPaperTradingError, Order
from agentfund.broker.alpaca import LIVE_TRADING_URL
from agentfund.data.alpaca import alpaca_store
from agentfund.paper import PaperTrader, plan_orders
from agentfund.audit import DecisionLog, verify
from agentfund.strategies import EqualWeight

KEYS = dict(key_id="PKTEST", secret="secret")


def fake_alpaca(account_number="PA1234567", positions=None, symbols=("AAPL", "MSFT", "JPM", "XOM")):
    sent = []
    days = pd.bdate_range("2024-01-02", periods=40)

    def handler(req: httpx.Request) -> httpx.Response:
        host, path = req.url.host, req.url.path
        if host == "data.alpaca.markets" and path == "/v2/stocks/bars":
            bars = {
                s: [{"t": f"{d.date()}T05:00:00Z", "o": 100 + i, "h": 101 + i, "l": 99 + i, "c": 100 + i + k, "v": 5e6}
                    for i, d in enumerate(days)]
                for k, s in enumerate(req.url.params["symbols"].split(","))
            }
            return httpx.Response(200, json={"bars": bars, "next_page_token": None})
        assert host == "paper-api.alpaca.markets", f"unexpected host {host}"
        if path == "/v2/account":
            return httpx.Response(200, json={"account_number": account_number, "equity": "100000", "status": "ACTIVE"})
        if path == "/v2/positions":
            return httpx.Response(200, json=positions or [])
        if path == "/v2/orders" and req.method == "GET":
            return httpx.Response(200, json=[])
        if path == "/v2/orders" and req.method == "POST":
            body = json.loads(req.content)
            sent.append(body)
            return httpx.Response(200, json={"id": f"ord{len(sent)}", "status": "accepted", **body})
        return httpx.Response(404)

    return httpx.MockTransport(handler), sent, {s: (s, sec) for s, sec in zip(symbols, ["Tech", "Tech", "Fin", "Energy"])}


def test_refuses_live_trading_url():
    with pytest.raises(NotPaperTradingError):
        AlpacaPaperClient(**KEYS, base_url=LIVE_TRADING_URL)


def test_refuses_live_url_from_environment(monkeypatch):
    monkeypatch.setenv("APCA_API_BASE_URL", LIVE_TRADING_URL)
    with pytest.raises(NotPaperTradingError):
        AlpacaPaperClient(**KEYS)


def test_refuses_non_paper_account():
    transport, sent, _ = fake_alpaca(account_number="123456789")
    client = AlpacaPaperClient(**KEYS, transport=transport)
    with pytest.raises(NotPaperTradingError):
        client.connect()
    with pytest.raises(NotPaperTradingError):
        client.submit(Order("AAPL", "buy", notional=100))
    assert sent == []


def test_client_has_no_money_movement_methods():
    names = {n.lower() for n in dir(AlpacaPaperClient)}
    for word in ("transfer", "bank", "ach", "wire", "fund", "deposit", "withdraw"):
        assert not any(word in n for n in names), word


def test_plan_orders_sells_first_and_skips_tiny_trades():
    positions = {"AAPL": {"qty": "10", "market_value": "2000", "current_price": "200"},
                 "OLD": {"qty": "5", "market_value": "500", "current_price": "100"}}
    orders = plan_orders({"AAPL": 0.0204, "MSFT": 0.05}, positions, 100_000, {"AAPL": 200, "MSFT": 400})
    assert [(o.symbol, o.side) for o in orders] == [("OLD", "sell"), ("MSFT", "buy")]
    assert orders[0].qty == 5 and orders[1].notional == 5000


def test_full_cycle_dry_run_then_submit(tmp_path):
    transport, sent, syms = fake_alpaca()
    store = alpaca_store(syms, start="2024-01-01", end="2024-03-01", transport=transport, **KEYS)
    assert set(store.prices.ticker) == set(syms)
    client = AlpacaPaperClient(**KEYS, transport=transport)
    log = DecisionLog(tmp_path / "d.jsonl")
    trader = PaperTrader(client, store, EqualWeight(), log=log, state_path=tmp_path / "s.json")

    dry = trader.run_cycle(submit=False)
    assert dry.orders and sent == []

    live = trader.run_cycle(submit=True)
    assert len(sent) == len(live.orders) > 0
    assert all(o["type"] == "market" for o in sent)
    # Risk limits still apply: no single name above 10% of equity.
    assert all(float(o["notional"]) <= 10_000 + 1e-6 for o in sent if o["side"] == "buy")
    assert verify(log.path)[0]
