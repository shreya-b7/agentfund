"""Persistence: DuckDB file and CSV loaders for :class:`PointInTimeStore`."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .pit import PointInTimeStore

TABLES = ("prices", "universe", "fundamentals")


def save_duckdb(store: PointInTimeStore, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as con:
        for name in TABLES:
            df = getattr(store, name)  # noqa: F841 - referenced by DuckDB replacement scan
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")


def load_duckdb(path: str | Path) -> PointInTimeStore:
    with duckdb.connect(str(path), read_only=True) as con:
        frames = {name: con.execute(f"SELECT * FROM {name}").df() for name in TABLES}
    return PointInTimeStore(**frames)


def load_csv_dir(path: str | Path) -> PointInTimeStore:
    """Load ``prices.csv``, ``universe.csv`` and (optional) ``fundamentals.csv`` from a folder.

    Use this to plug in real data. ``universe.csv`` must include delisted names with
    their ``end_date``; leaving them out reintroduces survivorship bias.
    """
    path = Path(path)
    prices = pd.read_csv(path / "prices.csv")
    universe = pd.read_csv(path / "universe.csv")
    fpath = path / "fundamentals.csv"
    fundamentals = (
        pd.read_csv(fpath) if fpath.exists() else pd.DataFrame(columns=["ticker", "period_end", "filed_date"])
    )
    return PointInTimeStore(prices=prices, universe=universe, fundamentals=fundamentals)
