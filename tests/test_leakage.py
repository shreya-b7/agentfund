"""CI guard: these tests fail if look-ahead bias is introduced anywhere in the decision path."""

import pandas as pd

from agentfund.agents import AnalystAgent, OfflineLLM
from agentfund.agents.analyst import offline_analyst_responder
from agentfund.backtest.leakage import check_backtest, check_strategy
from agentfund.strategies import STRATEGIES, AnalystStrategy, Momentum

CUTOFF = "2019-03-29"


class PeekingMomentum(Momentum):
    """Deliberately leaky: ranks on the *next* month's return using the raw store."""

    name = "peeking"

    def __init__(self, store):
        super().__init__()
        self._close = store.wide("close")

    def weights(self, view):
        nxt = self._close.loc[view.as_of :].iloc[:22]
        fwd = (nxt.iloc[-1] / nxt.iloc[0] - 1).reindex(view.tickers()).dropna()
        top = fwd.sort_values(ascending=False).index[:5]
        return {t: 0.2 for t in top}


def test_harness_catches_a_leaky_strategy(store):
    rep = check_strategy(PeekingMomentum, store, CUTOFF)
    assert not rep.passed


def test_all_baselines_are_leak_free(store):
    for name, cls in STRATEGIES.items():
        rep = check_strategy(lambda s, cls=cls: cls(), store, CUTOFF)
        assert rep.passed, f"{name}: {rep.detail}"


def test_backtest_equity_is_leak_free(store):
    rep = check_backtest(lambda s: Momentum(), store, CUTOFF)
    assert rep.passed, rep.detail


def test_analyst_strategy_is_leak_free(store):
    def factory(s):
        llm = OfflineLLM(offline_analyst_responder)
        return AnalystStrategy(AnalystAgent(llm), coverage=5)

    rep = check_strategy(factory, store, CUTOFF, start="2018-10-01")
    assert rep.passed, rep.detail


def test_orders_fill_at_next_open(store):
    from agentfund.backtest import Backtester

    r = Backtester(store, Momentum()).run(end="2018-06-29")
    first_decision = pd.Timestamp(r.decisions[0]["date"])
    first_fill = r.trades.loc[r.trades.reason == "rebalance", "date"].min()
    assert first_fill > first_decision
    opens = store.wide("open")
    t = r.trades[r.trades.reason == "rebalance"].iloc[0]
    assert t.price == opens.loc[t.date, t.ticker]


def test_portfolio_construction_is_leak_free(store):
    from agentfund.strategies import MinVariance, RiskParity

    for cls in (RiskParity, MinVariance):
        rep = check_strategy(lambda s, cls=cls: cls(), store, CUTOFF, start="2018-06-01")
        assert rep.passed, rep.detail
