import pandas as pd
import pytest

from agentfund.data import LookAheadError, PointInTimeStore
from agentfund.data.store import load_duckdb, save_duckdb


def test_view_refuses_future(store):
    v = store.view("2018-06-29")
    with pytest.raises(LookAheadError):
        v.history("close", end="2018-07-02")
    with pytest.raises(LookAheadError):
        v.universe(on="2019-01-02")
    assert v.history("close").index.max() <= pd.Timestamp("2018-06-29")


def test_fundamentals_use_filed_date_not_period_end(store):
    f = store.fundamentals
    row = f.iloc[0]
    # The day after the quarter ends but before filing, the quarter is unknown.
    before = store.view(row["filed_date"] - pd.Timedelta(days=1)).fundamentals(row["ticker"])
    assert not (before["period_end"] == row["period_end"]).any()
    after = store.view(row["filed_date"]).fundamentals(row["ticker"])
    assert (after["period_end"] == row["period_end"]).any()


def test_restatement_only_visible_after_refiling(store):
    f = store.fundamentals
    dup = f[f.duplicated(["ticker", "period_end"], keep=False)]
    assert not dup.empty, "synthetic data should include restatements"
    tk, pe = dup.iloc[0][["ticker", "period_end"]]
    rows = dup[(dup.ticker == tk) & (dup.period_end == pe)].sort_values("filed_date")
    original, restated = rows.iloc[0], rows.iloc[-1]
    mid = store.view(restated["filed_date"] - pd.Timedelta(days=1)).fundamentals(tk)
    assert mid.loc[mid.period_end == pe, "eps"].item() == pytest.approx(original["eps"])
    late = store.view(restated["filed_date"]).fundamentals(tk)
    assert late.loc[late.period_end == pe, "eps"].item() == pytest.approx(restated["eps"])


def test_universe_is_survivorship_free(store):
    delisted = store.universe.dropna(subset=["end_date"])
    assert not delisted.empty
    tk, end = delisted.iloc[0][["ticker", "end_date"]]
    last_trading_day = store.dates[store.dates < end][-1]
    assert tk in store.view(last_trading_day).tickers()
    assert tk not in store.view(end).tickers()
    # ...but its history is still there for earlier dates.
    assert tk in store.prices["ticker"].unique()


def test_rejects_filing_before_period_end():
    with pytest.raises(ValueError):
        PointInTimeStore(
            prices=pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"]),
            universe=pd.DataFrame(columns=["ticker", "name", "sector", "start_date", "end_date"]),
            fundamentals=pd.DataFrame({"ticker": ["A"], "period_end": ["2020-03-31"], "filed_date": ["2020-03-01"]}),
        )


def test_duckdb_roundtrip(store, tmp_path):
    save_duckdb(store, tmp_path / "m.duckdb")
    back = load_duckdb(tmp_path / "m.duckdb")
    pd.testing.assert_frame_equal(back.prices, store.prices, check_dtype=False)
    assert len(back.fundamentals) == len(store.fundamentals)
