"""
features.py
===========
Turns raw OHLCV prices into a feature matrix the models can learn from.

The central modeling choice is *what we predict* (the target) and *what we
predict it from* (the features). Two important rules drive everything here:

1. No look-ahead leakage. Every feature for day t must be computable using only
   information available up to and including day t. If a feature accidentally
   peeks at the future, the model looks brilliant in testing and useless in
   reality. This is the single most common way stock ML projects fool
   themselves.

2. The target is tomorrow's close. We line up each row's features (known today)
   against the *next* day's closing price, so the model learns to predict one
   step ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price DataFrame (Open/High/Low/Close/Volume), return a new
    DataFrame with engineered features plus the prediction target.

    Adds 10+ time-series features: lagged returns, rolling averages,
    rolling volatility, momentum, and volume signals. All are computed
    from past data only.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    # --- Daily return: percent change vs yesterday. The most basic signal. ---
    out["return_1d"] = close.pct_change(1)

    # --- Lagged returns: returns from a few days back, so the model can see
    #     short-term patterns/momentum. ---
    for lag in (2, 3, 5):
        out[f"return_{lag}d"] = close.pct_change(lag)

    # --- Rolling means (simple moving averages) and the price's distance from
    #     them. "How far is today's price above/below its recent average?" ---
    for window in (5, 10, 20):
        sma = close.rolling(window).mean()
        out[f"sma_{window}"] = sma
        out[f"close_to_sma_{window}"] = close / sma - 1.0  # relative gap

    # --- Rolling volatility: std-dev of daily returns over a window. Captures
    #     how choppy the stock has been recently. ---
    daily_ret = close.pct_change(1)
    for window in (5, 10):
        out[f"volatility_{window}"] = daily_ret.rolling(window).std()

    # --- Momentum: total return over the last N days. ---
    out["momentum_10"] = close / close.shift(10) - 1.0

    # --- Volume signal: today's volume relative to its 10-day average. Spikes
    #     in volume often accompany meaningful moves. ---
    vol = df["Volume"]
    out["volume_ratio_10"] = vol / vol.rolling(10).mean()

    # --- High-low range as a fraction of close: an intraday volatility proxy. ---
    out["hl_range"] = (df["High"] - df["Low"]) / close

    # ------------------------------------------------------------------ #
    # Targets
    # ------------------------------------------------------------------ #
    # Regression target: tomorrow's closing price (shift target back by -1 so
    # today's features align with the next day's outcome).
    out["target_close"] = close.shift(-1)

    # Classification-style target for "directional accuracy": did the price go
    # up the next day? 1 if tomorrow's close > today's close, else 0.
    out["target_up"] = (close.shift(-1) > close).astype(int)

    # Drop rows with NaNs introduced by rolling windows (start) and the shift
    # (last row has no "tomorrow").
    out = out.dropna()
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return just the predictor columns (everything except the targets)."""
    return [c for c in df.columns if not c.startswith("target_")]
