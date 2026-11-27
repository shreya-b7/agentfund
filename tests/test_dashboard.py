import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from agentfund.audit import DecisionLog
from agentfund.dashboard.server import DashboardState, make_handler


def test_dashboard_endpoints(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    log.append("paper_cycle", {"as_of": "2024-01-02", "risk": {"verdict": "approved"}, "orders": []})
    state = DashboardState(str(tmp_path / "none.duckdb"), str(tmp_path / "acct.json"), str(log.path))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "PAPER TRADING" in page
        s = json.loads(urllib.request.urlopen(base + "/api/summary").read())
        assert s["account"] is None and s["log"]["ok"] and s["decisions"][0]["kind"] == "paper_cycle"
        for _ in range(120):  # backtests run in the background (synthetic data here)
            bt = json.loads(urllib.request.urlopen(base + "/api/backtests").read())
            if bt["status"] != "computing":
                break
            time.sleep(0.5)
        assert bt["status"] == "ready" and "risk_parity" in bt["table"]
        assert not (tmp_path / "acct.json").exists()  # read-only: never creates or changes the account
    finally:
        httpd.shutdown()
