import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from agentfund.backtest.attribution import _read_zip, attribute, parse_french_csv
from agentfund.backtest.walkforward import walk_forward
from agentfund.strategies import STRATEGIES, EqualWeight, Momentum, MinVariance, RiskParity
from agentfund.strategies.portfolio import ledoit_wolf, min_variance, risk_contributions, risk_parity

SAMPLE = """This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 202509 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.02,   -0.35,    0.03,    0.13,    0.012
19630702,    0.79,   -0.28,    0.28,   -0.08,   -0.21,    0.012
19630703,    0.63,   -0.18,   -0.10,    0.13,   -0.25,    0.012
"""


def test_parse_french_csv_and_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F.csv", SAMPLE)
    df = parse_french_csv(_read_zip(buf.getvalue()))
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 3 and df.loc["1963-07-02", "Mkt-RF"] == pytest.approx(0.0079)


def test_attribution_recovers_known_betas():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    f = pd.DataFrame(rng.normal(0, 0.01, (1500, 6)), index=idx, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    f["RF"] = 0.0001
    true = {"Mkt-RF": 1.1, "SMB": 0.3, "HML": -0.2, "RMW": 0.0, "CMA": 0.1, "Mom": 0.4}
    alpha = 0.0004
    r = f["RF"] + alpha + sum(b * f[c] for c, b in true.items()) + rng.normal(0, 0.002, 1500)
    out = attribute(r, f)
    for c, b in true.items():
        assert out["betas"][c] == pytest.approx(b, abs=0.03)
    assert out["alpha_annual"] == pytest.approx(alpha * 252, abs=0.03)
    assert out["alpha_t"] > 2 and out["r2"] > 0.9


def test_ledoit_wolf_shrinks_noisy_covariance():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, (60, 40))  # few observations, many names: sample cov is very noisy
    cov, shrink = ledoit_wolf(x)
    assert 0 < shrink <= 1
    assert np.linalg.eigvalsh(cov).min() > 0  # well conditioned, unlike the sample covariance


def test_min_variance_respects_cap_and_beats_equal_weight():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (500, 20)) * np.linspace(0.005, 0.03, 20)
    cov, _ = ledoit_wolf(x)
    w = min_variance(cov, cap=0.10)
    ew = np.full(20, 1 / 20)
    assert w.sum() == pytest.approx(1) and w.min() >= -1e-12 and w.max() <= 0.10 + 1e-9
    assert w @ cov @ w < ew @ cov @ ew


def test_risk_parity_equalizes_risk_contributions():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (500, 10)) * np.linspace(0.005, 0.03, 10)
    cov, _ = ledoit_wolf(x)
    rc = risk_contributions(risk_parity(cov), cov)
    assert np.allclose(rc, 0.1, atol=1e-6)


def test_new_strategies_registered_and_run(store):
    from agentfund.backtest import Backtester

    assert {"risk_parity", "min_variance"} <= set(STRATEGIES)
    for cls in (RiskParity, MinVariance):
        r = Backtester(store, cls()).run(end="2018-12-31")
        assert all(max(d["approved"].values(), default=0) <= 0.10 + 1e-9 for d in r.decisions)


def test_walk_forward_only_uses_the_past(store):
    variants = {"ew": EqualWeight, "mom": Momentum}
    wf = walk_forward(store, variants, train_days=252, test_days=126)
    assert len(wf.choices) >= 3
    first_test = pd.Timestamp(wf.choices.test_start.iloc[0])
    assert first_test >= store.dates[252]
    assert wf.oos_returns.index.is_monotonic_increasing and not wf.oos_returns.index.duplicated().any()
    assert set(wf.choices.chosen) <= set(variants)
    s = wf.summary()
    assert {"oos_sharpe", "oos_deflated_sharpe", "hindsight_best_sharpe"} <= set(s)


def test_risk_parity_handles_negative_correlation():
    cov = np.array([[0.04, 0.03, -0.01], [0.03, 0.04, -0.01], [-0.01, -0.01, 0.01]])
    w = risk_parity(cov)
    assert np.isfinite(w).all() and (w > 0).all() and w.sum() == pytest.approx(1)
    assert np.allclose(risk_contributions(w, cov), 1 / 3, atol=1e-6)


def test_risk_parity_never_returns_an_empty_portfolio_on_hedged_universes():
    rng = np.random.default_rng(5)
    for _ in range(50):
        f = rng.normal(0, 0.01, (300, 1))
        loads = rng.uniform(-1, 1, (1, 12))
        x = f @ loads + rng.normal(0, 0.005, (300, 12))
        cov, _ = ledoit_wolf(x)
        w = risk_parity(cov)
        assert np.isfinite(w).all() and (w > 0).all()
