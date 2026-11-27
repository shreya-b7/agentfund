"""Local dashboard: ``python -m agentfund dashboard`` then open http://localhost:8050.

Read-only. It shows the paper account, the decision log (streamed as new entries
arrive, via Server-Sent Events) and a backtest comparison of every strategy. It
never places orders or changes the account.

Uses only the Python standard library for serving; the page loads Chart.js from a CDN.
"""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from ..audit import verify
from ..backtest import Backtester, summarize
from ..data.store import load_duckdb
from ..data.synthetic import generate_market
from ..strategies import STRATEGIES
from .page import PAGE


def _clean(obj):
    """Make an object JSON-safe (NaN/inf -> None, timestamps -> strings)."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


class DashboardState:
    def __init__(self, prices: str, account: str, log: str):
        self.prices_path, self.account_path, self.log_path = Path(prices), Path(account), Path(log)
        self.backtests: dict | None = None
        self.backtest_error: str | None = None
        threading.Thread(target=self._compute_backtests, daemon=True).start()

    def _store(self):
        if self.prices_path.exists():
            return load_duckdb(self.prices_path), f"prices from {self.prices_path}"
        return generate_market(), "synthetic demo data (run fetch for real prices)"

    def _compute_backtests(self) -> None:
        try:
            store, source = self._store()
            curves, table = {}, {}
            for name, cls in STRATEGIES.items():
                res = Backtester(store, cls()).run()
                weekly = res.equity.resample("W-FRI").last().dropna() / res.equity.iloc[0]
                curves[name] = {"dates": [str(d.date()) for d in weekly.index], "values": weekly.round(4).tolist()}
                table[name] = summarize(res, n_trials=len(STRATEGIES))
            self.backtests = _clean({"source": source, "curves": curves, "table": table})
        except Exception as e:  # shown on the page instead of crashing the server
            self.backtest_error = f"{type(e).__name__}: {e}"

    def summary(self) -> dict:
        out: dict = {"account": None, "positions": [], "pending": 0, "equity_history": [], "prices_to": None}
        if self.account_path.exists():
            state = json.loads(self.account_path.read_text())
            out["equity_history"] = state.get("equity_history", [])
            out["pending"] = len(state.get("pending", []))
            out["fills"] = state.get("fills", [])[-20:]
            if self.prices_path.exists():
                from ..paper.local_broker import LocalPaperBroker

                broker = LocalPaperBroker(load_duckdb(self.prices_path), self.account_path)  # read only: no connect()
                out["account"] = broker.account()
                out["positions"] = broker.positions()
                out["prices_to"] = str(broker.store.dates[-1].date())
        paper_dir = self.account_path.parent
        cfg_path = paper_dir / "config.json"
        out["strategy"] = json.loads(cfg_path.read_text()).get("strategy") if cfg_path.exists() else None
        out["paused"] = (paper_dir / "PAUSED").exists()
        out["log"] = self.log_status()
        out["decisions"] = self.decisions(limit=400)
        return _clean(out)

    def log_status(self) -> dict:
        if not self.log_path.exists():
            return {"exists": False}
        ok, msg = verify(self.log_path)
        return {"exists": True, "ok": ok, "message": msg}

    def decisions(self, limit: int = 25) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text().splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()][::-1]


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/summary":
                self._json(state.summary())
            elif path == "/api/backtests":
                if state.backtest_error:
                    self._json({"status": "error", "error": state.backtest_error})
                elif state.backtests is None:
                    self._json({"status": "computing"})
                else:
                    self._json({"status": "ready", **state.backtests})
            elif path == "/api/stream":
                self._stream()
            else:
                self._send(b"not found", "text/plain", 404)

        def _stream(self) -> None:
            """Server-Sent Events: push each new decision-log record as it is written."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = len(state.log_path.read_text().splitlines()) if state.log_path.exists() else 0
            try:
                while True:
                    lines = state.log_path.read_text().splitlines() if state.log_path.exists() else []
                    for line in lines[seen:]:
                        if line.strip():
                            self.wfile.write(f"event: decision\ndata: {line}\n\n".encode())
                    seen = len(lines)
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    time.sleep(2)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def serve(
    port: int = 8050,
    prices: str = "data/prices.duckdb",
    account: str = "runs/paper/local_account.json",
    log: str = "runs/paper/decisions.jsonl",
) -> None:
    state = DashboardState(prices, account, log)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    httpd.daemon_threads = True
    print(f"dashboard running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
