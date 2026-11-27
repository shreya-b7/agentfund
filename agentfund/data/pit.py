"""Point-in-time market data.

Every record carries the time at which it became *knowable*:

* prices: a bar dated ``t`` is known after the close of ``t``.
* fundamentals: a value is known from its ``filed_date`` (not its ``period_end``).
  Restatements are separate rows with a later ``filed_date``; a view returns the
  latest filing known at ``as_of``.
* universe: a ticker is a member while ``start_date <= t < end_date``. Delisted
  names stay in history, so backtests are free of survivorship bias.

Strategies and agents never touch :class:`PointInTimeStore` directly. They get a
:class:`PITView` frozen at a decision time, which refuses any request for data
from after that time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
UNIVERSE_COLUMNS = ["ticker", "name", "sector", "start_date", "end_date"]
FUNDAMENTAL_KEYS = ["ticker", "period_end", "filed_date"]


class LookAheadError(RuntimeError):
    """Raised when code asks for data that was not knowable at the view's as-of time."""


def _ts(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


@dataclass
class PointInTimeStore:
    """Holds the full history. Only hand out :class:`PITView` objects to decision code."""

    prices: pd.DataFrame
    universe: pd.DataFrame
    fundamentals: pd.DataFrame

    def __post_init__(self) -> None:
        self.prices = self._clean_prices(self.prices)
        self.universe = self._clean_universe(self.universe)
        self.fundamentals = self._clean_fundamentals(self.fundamentals)
        self._dates = pd.DatetimeIndex(sorted(self.prices["date"].unique()))
        self._wide_cache: dict[str, pd.DataFrame] = {}

    # ----- validation -------------------------------------------------
    @staticmethod
    def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
        missing = set(PRICE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"prices missing columns: {sorted(missing)}")
        out = df[PRICE_COLUMNS].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        if out.duplicated(["date", "ticker"]).any():
            raise ValueError("prices contain duplicate (date, ticker) rows")
        return out.sort_values(["date", "ticker"]).reset_index(drop=True)

    @staticmethod
    def _clean_universe(df: pd.DataFrame) -> pd.DataFrame:
        missing = set(UNIVERSE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"universe missing columns: {sorted(missing)}")
        out = df[UNIVERSE_COLUMNS].copy()
        out["start_date"] = pd.to_datetime(out["start_date"]).dt.normalize()
        out["end_date"] = pd.to_datetime(out["end_date"]).dt.normalize()  # NaT = still listed
        return out.reset_index(drop=True)

    @staticmethod
    def _clean_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
        missing = set(FUNDAMENTAL_KEYS) - set(df.columns)
        if missing:
            raise ValueError(f"fundamentals missing columns: {sorted(missing)}")
        out = df.copy()
        out["period_end"] = pd.to_datetime(out["period_end"]).dt.normalize()
        out["filed_date"] = pd.to_datetime(out["filed_date"]).dt.normalize()
        if (out["filed_date"] < out["period_end"]).any():
            raise ValueError("a fundamental row is filed before its period ended")
        return out.sort_values(["ticker", "period_end", "filed_date"]).reset_index(drop=True)

    # ----- accessors --------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    def wide(self, field: str) -> pd.DataFrame:
        """Date x ticker matrix of one price field (full history; engine use only)."""
        if field not in self._wide_cache:
            self._wide_cache[field] = self.prices.pivot(index="date", columns="ticker", values=field)
        return self._wide_cache[field]

    def view(self, as_of) -> "PITView":
        return PITView(self, _ts(as_of))


class PITView:
    """Read-only window onto a :class:`PointInTimeStore` as of one date (after its close)."""

    __slots__ = ("_store", "_as_of")

    def __init__(self, store: PointInTimeStore, as_of: pd.Timestamp):
        self._store = store
        self._as_of = as_of

    @property
    def as_of(self) -> pd.Timestamp:
        return self._as_of

    def _check(self, when, what: str) -> pd.Timestamp:
        when = _ts(when)
        if when > self._as_of:
            raise LookAheadError(
                f"{what} requested for {when.date()} but this view is as of {self._as_of.date()}"
            )
        return when

    def universe(self, on=None) -> pd.DataFrame:
        """Members listed on ``on`` (default: as_of). Delisted names are excluded after delisting."""
        on = self._check(on if on is not None else self._as_of, "universe")
        u = self._store.universe
        live = (u["start_date"] <= on) & (u["end_date"].isna() | (u["end_date"] > on))
        return u.loc[live, ["ticker", "name", "sector"]].reset_index(drop=True)

    def tickers(self) -> list[str]:
        return self.universe()["ticker"].tolist()

    def sectors(self) -> dict[str, str]:
        u = self.universe()
        return dict(zip(u["ticker"], u["sector"]))

    def history(self, field: str = "close", lookback: int | None = None, end=None) -> pd.DataFrame:
        """Date x ticker matrix up to ``end`` (default: as_of), last ``lookback`` rows."""
        end = self._check(end if end is not None else self._as_of, f"{field} history")
        wide = self._store.wide(field)
        out = wide.loc[:end]
        if lookback is not None:
            out = out.iloc[-lookback:]
        return out.copy()

    def latest(self, field: str = "close") -> pd.Series:
        return self.history(field, lookback=1).iloc[-1]

    def fundamentals(self, ticker: str | None = None) -> pd.DataFrame:
        """Latest known filing per (ticker, period_end), using only filings dated <= as_of."""
        f = self._store.fundamentals
        known = f[f["filed_date"] <= self._as_of]
        if ticker is not None:
            known = known[known["ticker"] == ticker]
        latest = known.groupby(["ticker", "period_end"], as_index=False).tail(1)
        return latest.reset_index(drop=True).copy()
