import pytest

from agentfund.risk import RiskContext, RiskLimits, RiskOfficer, Verdict


def ctx(**kw):
    base = dict(
        current_weights={},
        prices={"A": 50, "B": 50, "C": 50, "D": 2, "E": 50},
        adv_dollars={k: 1e9 for k in "ABCDE"},
        sectors={"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy", "E": "Health"},
        tradeable=set("ABCDE"),
        equity=1e6,
        drawdown=0.0,
    )
    base.update(kw)
    return RiskContext(**base)


def test_approves_clean_proposal():
    d = RiskOfficer().review({"A": 0.1, "E": 0.1}, ctx())
    assert d.verdict is Verdict.APPROVED and d.approved == {"A": 0.1, "E": 0.1}


def test_position_and_sector_caps():
    d = RiskOfficer().review({"A": 0.2, "B": 0.1, "C": 0.1, "E": 0.05}, ctx())
    assert d.verdict is Verdict.RESIZED
    assert max(d.approved.values()) <= 0.10 + 1e-9
    tech = sum(v for k, v in d.approved.items() if k in "ABC")
    assert tech <= 0.30 + 1e-9
    assert {b.rule for b in d.breaches} >= {"max_position_weight"}


def test_long_only_min_price_and_untradeable():
    d = RiskOfficer().review({"A": -0.05, "D": 0.05, "Z": 0.05, "E": 0.05}, ctx())
    rules = {b.rule for b in d.breaches}
    assert {"long_only", "min_price", "not_tradeable"} <= rules
    assert d.approved == {"E": 0.05}


def test_liquidity_cap():
    d = RiskOfficer().review({"E": 0.1}, ctx(adv_dollars={"E": 1e6}))  # 5% of $1m = $50k = 5% of equity
    assert d.approved["E"] == pytest.approx(0.05)


def test_turnover_cap_blends_toward_current():
    lim = RiskLimits(max_turnover=0.1, max_sector_weight=1.0)
    d = RiskOfficer(lim).review({"A": 0.1, "E": 0.1}, ctx(current_weights={"B": 0.1, "C": 0.1}))
    turnover = sum(abs(d.approved.get(k, 0) - {"B": 0.1, "C": 0.1}.get(k, 0)) for k in "ABCE") / 2
    assert turnover == pytest.approx(0.1)


def test_kill_switch_flattens():
    d = RiskOfficer().review({"A": 0.1}, ctx(drawdown=0.3))
    assert d.verdict is Verdict.VETOED and d.approved == {}


def test_limits_are_frozen_and_only_tighten():
    lim = RiskLimits()
    with pytest.raises(Exception):
        lim.max_position_weight = 0.5  # type: ignore[misc]
    assert lim.tightened(max_position_weight=0.05).max_position_weight == 0.05
    with pytest.raises(PermissionError):
        lim.tightened(max_position_weight=0.5)
    with pytest.raises(PermissionError):
        lim.tightened(min_price=1.0)
    with pytest.raises(PermissionError):
        lim.tightened(long_only=False)
    assert lim.fingerprint() != lim.tightened(max_gross_exposure=0.9).fingerprint()
