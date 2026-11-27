import numpy as np
import pandas as pd
import pytest

from agentfund.audit import DecisionLog, verify
from agentfund.backtest import Backtester, CostModel, summarize
from agentfund.backtest.metrics import deflated_sharpe, probabilistic_sharpe
from agentfund.strategies import EqualWeight


def test_costs_reduce_returns(store):
    free = Backtester(store, EqualWeight(), costs=CostModel(0, 0, 0)).run()
    paid = Backtester(store, EqualWeight(), costs=CostModel(5, 10, 0.2)).run()
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.total_cost > 0 and free.total_cost == 0


def test_every_decision_is_logged_and_verifiable(store, tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    r = Backtester(store, EqualWeight(), log=log).run(end="2017-12-29")
    assert len(log.records()) == len(r.decisions) > 0
    assert verify(log.path)[0]


def test_approved_weights_respect_limits(store):
    r = Backtester(store, EqualWeight(max_names=5)).run(end="2018-12-31")
    for d in r.decisions:
        assert all(v <= 0.10 + 1e-9 for v in d["approved"].values())


def test_summary_fields(store):
    s = summarize(Backtester(store, EqualWeight()).run(), n_trials=10)
    assert {"sharpe", "max_drawdown", "deflated_sharpe", "psr"} <= set(s)
    assert 0 <= s["deflated_sharpe"] <= s["psr"] <= 1


def test_deflated_sharpe_falls_with_more_trials():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    vals = [deflated_sharpe(r, n) for n in (1, 10, 100, 1000)]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] == pytest.approx(probabilistic_sharpe(r, 0.0))
