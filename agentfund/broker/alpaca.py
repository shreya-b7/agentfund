"""Alpaca **paper trading** client. Paper only, by construction.

Safety design:

* The trading URL is a constant: Alpaca's paper endpoint. There is no setting,
  argument or environment variable that changes it, and the constructor refuses
  any other base URL (including Alpaca's live endpoint).
* :meth:`AlpacaPaperClient.connect` checks the account Alpaca returns is a paper
  account (paper account numbers start with ``PA``) and refuses to continue if
  not.
* The client only implements account/positions/orders/clock calls. It has no
  code for bank links, transfers, funding or anything that moves real money.

Paper accounts trade with simulated money against real, current market prices.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import httpx

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_URL = "https://api.alpaca.markets"  # listed only so we can refuse it
DATA_URL = "https://data.alpaca.markets"


class NotPaperTradingError(RuntimeError):
    """Raised when anything suggests we are not talking to a paper account."""


def _keys(key_id: str | None, secret: str | None) -> tuple[str, str]:
    key_id = key_id or os.environ.get("ALPACA_API_KEY_ID")
    secret = secret or os.environ.get("ALPACA_API_SECRET_KEY")
    if not key_id or not secret:
        raise RuntimeError("set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (paper keys)")
    return key_id, secret


def _headers(key_id: str, secret: str) -> dict[str, str]:
    return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    notional: float | None = None  # dollar amount (fractional), for buys
    qty: float | None = None  # shares, for sells
    client_order_id: str = ""


class AlpacaPaperClient:
    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        base_url: str = PAPER_TRADING_URL,
        transport: httpx.BaseTransport | None = None,
    ):
        if base_url.rstrip("/") != PAPER_TRADING_URL:
            raise NotPaperTradingError(f"refusing trading URL {base_url!r}; only {PAPER_TRADING_URL} is allowed")
        if os.environ.get("APCA_API_BASE_URL", PAPER_TRADING_URL).rstrip("/") != PAPER_TRADING_URL:
            raise NotPaperTradingError("APCA_API_BASE_URL points somewhere other than the paper endpoint")
        kid, sec = _keys(key_id, secret)
        self._http = httpx.Client(base_url=PAPER_TRADING_URL, headers=_headers(kid, sec), timeout=30, transport=transport)
        self._verified = False

    # ----- safety -----------------------------------------------------
    def connect(self) -> dict:
        acct = self._get("/v2/account", _skip_check=True)
        number = str(acct.get("account_number", ""))
        if not number.startswith("PA"):
            raise NotPaperTradingError(f"account {number!r} is not a paper account (expected a 'PA' prefix)")
        self._verified = True
        return acct

    def _require_paper(self) -> None:
        if not self._verified:
            self.connect()

    # ----- http -------------------------------------------------------
    def _get(self, path: str, params: dict | None = None, _skip_check: bool = False):
        if not _skip_check:
            self._require_paper()
        r = self._http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        self._require_paper()
        r = self._http.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca rejected order: {r.status_code} {r.text}")
        return r.json()

    # ----- read -------------------------------------------------------
    def account(self) -> dict:
        return self._get("/v2/account")

    def positions(self) -> list[dict]:
        return self._get("/v2/positions")

    def clock(self) -> dict:
        return self._get("/v2/clock")

    def open_orders(self) -> list[dict]:
        return self._get("/v2/orders", {"status": "open"})

    # ----- trade (paper) ----------------------------------------------
    def submit(self, order: Order) -> dict:
        body = {
            "symbol": order.symbol,
            "side": order.side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": order.client_order_id or f"af-{uuid.uuid4().hex[:20]}",
        }
        if order.notional is not None:
            body["notional"] = f"{order.notional:.2f}"
        else:
            body["qty"] = f"{order.qty:.6f}".rstrip("0").rstrip(".")
        return self._post("/v2/orders", body)
