"""Fama-French factor attribution.

Regresses a strategy's daily excess returns on the Fama-French five factors
(market, size, value, profitability, investment) plus momentum:

    r - rf = alpha + b1 MKT + b2 SMB + b3 HML + b4 RMW + b5 CMA + b6 MOM + e

Alpha is the return the factors don't explain. t-statistics use Newey-West
standard errors, which allow for autocorrelated, heteroskedastic residuals.
Factor data comes free from Kenneth French's data library.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_FILE = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_FILE = "F-F_Momentum_Factor_daily_CSV.zip"
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French CSV: free-text header, then ``YYYYMMDD,v1,v2,...`` rows in percent."""
    lines = text.splitlines()
    header, rows = None, []
    for i, line in enumerate(lines):
        first = line.split(",")[0].strip()
        if len(first) == 8 and first.isdigit():
            if header is None:
                header = [h.strip() for h in lines[i - 1].split(",")]
            rows.append([c.strip() for c in line.split(",")])
        elif header is not None and rows:
            break  # daily block ended (annual tables etc. follow in some files)
    if header is None:
        raise ValueError("no daily rows found in factor file")
    df = pd.DataFrame(rows, columns=["date", *header[1:]])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={"Mom   ": "Mom", "MOM": "Mom", "Mom ": "Mom"})


def _read_zip(blob: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def download_factors(out: str | Path = "data/ff_factors.csv") -> pd.DataFrame:
    import httpx

    parts = []
    with httpx.Client(timeout=60, follow_redirects=True) as http:
        for name in (FF5_FILE, MOM_FILE):
            r = http.get(FF_BASE + name)
            r.raise_for_status()
            parts.append(parse_french_csv(_read_zip(r.content)))
    df = parts[0].join(parts[1], how="inner")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return df


def load_factors(path: str | Path = "data/ff_factors.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _newey_west_se(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    n = len(x)
    xtx_inv = np.linalg.inv(x.T @ x)
    u = x * resid[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = u[lag:].T @ u[:-lag]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv * n / (n - x.shape[1])
    return np.sqrt(np.diag(cov))


def attribute(returns: pd.Series, factors: pd.DataFrame, lags: int = 5) -> dict:
    cols = [c for c in FACTORS if c in factors.columns]
    data = pd.concat([returns.rename("r"), factors[cols + ["RF"]]], axis=1, join="inner").dropna()
    if len(data) < 60:
        raise ValueError("fewer than 60 overlapping days between returns and factors")
    y = (data["r"] - data["RF"]).to_numpy()
    x = np.column_stack([np.ones(len(data)), data[cols].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    se = _newey_west_se(x, resid, lags)
    r2 = 1 - resid.var() / y.var()
    return {
        "alpha_annual": float(beta[0] * 252),
        "alpha_t": float(beta[0] / se[0]),
        "betas": {c: float(b) for c, b in zip(cols, beta[1:])},
        "t_stats": {c: float(b / s) for c, b, s in zip(cols, beta[1:], se[1:])},
        "r2": float(r2),
        "days": int(len(data)),
        "start": str(data.index[0].date()),
        "end": str(data.index[-1].date()),
    }
