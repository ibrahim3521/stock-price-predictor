"""
data.py
=======
Responsible for getting raw price data into the pipeline.

Design decision: yfinance scrapes Yahoo Finance and can rate-limit or fail
intermittently. To make the project reproducible and offline-friendly, we
download once and cache to a CSV. Every run after the first reads the cache,
so you can iterate on features and models without re-hitting Yahoo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Directory where cached CSVs live (project_root/data).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_prices(
    ticker: str = "AAPL",
    period: str = "5y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of historical OHLCV prices for `ticker`.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. "AAPL".
    period : str
        How far back to pull, e.g. "5y" for five years.
    interval : str
        Bar size, e.g. "1d" for daily.
    force_refresh : bool
        If True, ignore any cached CSV and re-download from Yahoo.

    Returns
    -------
    pd.DataFrame
        Indexed by date, with columns Open, High, Low, Close, Volume.
    """
    cache_path = DATA_DIR / f"{ticker}_{period}_{interval}.csv"

    # Fast path: load from cache unless the caller forces a refresh.
    if cache_path.exists() and not force_refresh:
        print(f"[data] Loading cached prices from {cache_path.name}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return _clean(df)

    # Slow path: download from Yahoo Finance, then cache.
    print(f"[data] Downloading {ticker} ({period}, {interval}) from Yahoo Finance...")
    import yfinance as yf  # imported lazily so the cache path needs no network

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,   # adjust for splits/dividends
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Yahoo may be rate-limiting; "
            "try again in a minute, or use a cached CSV."
        )

    # yfinance sometimes returns a MultiIndex on columns for a single ticker;
    # flatten it so we always have simple column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    print(f"[data] Cached {len(df)} rows to {cache_path.name}")
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the standard OHLCV columns, drop rows with missing values."""
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df
