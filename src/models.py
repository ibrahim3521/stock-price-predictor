"""
models.py
=========
Trains the models and evaluates them honestly.

The most important idea here is the TIME-AWARE SPLIT. With time-series data you
must NOT shuffle rows into random train/test sets, because that lets the model
train on future data to predict the past, which is leakage. Instead we split
chronologically: train on the earliest ~80% of days, test on the most recent
~20%. That mimics reality, where you only ever have the past to predict the
future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import feature_columns


@dataclass
class Evaluation:
    """Container for one model's results, so the report code stays clean."""

    name: str
    rmse: float
    mae: float
    r2: float
    directional_accuracy: float


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Split a time-indexed DataFrame into (train, test) by time order.
    The last `test_frac` of rows become the test set.
    """
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _directional_accuracy(
    y_true_close: np.ndarray,
    y_pred_close: np.ndarray,
    today_close: np.ndarray,
) -> float:
    """
    Fraction of days where the model got the DIRECTION right: did it correctly
    predict whether tomorrow's close would be higher or lower than today's?

    This is the metric that actually matters for trading intuition, and the one
    that honestly exposes how hard the problem is (it tends to sit near 50%).
    """
    true_up = y_true_close > today_close
    pred_up = y_pred_close > today_close
    return float(np.mean(true_up == pred_up))


def evaluate_model(name, model, X_train, y_train, X_test, y_test, today_close_test):
    """Fit a model, predict on the test set, and compute all metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    dir_acc = _directional_accuracy(y_test.values, preds, today_close_test.values)

    return Evaluation(name, rmse, mae, r2, dir_acc), preds


def run_models(featured: pd.DataFrame):
    """
    Train Linear Regression and Random Forest plus a naive baseline, and return
    their evaluations.

    The naive baseline predicts "tomorrow's close = today's close." It is
    deliberately dumb, and it is a brutally strong benchmark for stock prices,
    which is exactly why it belongs here: any real model must be compared
    against it rather than against zero.
    """
    cols = feature_columns(featured)
    train, test = chronological_split(featured, test_frac=0.2)

    X_train, y_train = train[cols], train["target_close"]
    X_test, y_test = test[cols], test["target_close"]

    # "today's close" for each test row = the close we measure direction against.
    # close_to_sma_5 etc. are relative, so recover today's close from the data:
    today_close_test = test["sma_5"] * (1 + test["close_to_sma_5"])

    results = []

    # 1) Naive baseline: predict today's close as tomorrow's close.
    baseline_pred = today_close_test.values
    base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    base_mae = float(mean_absolute_error(y_test, baseline_pred))
    base_dir = _directional_accuracy(
        y_test.values, baseline_pred, today_close_test.values
    )
    results.append(
        Evaluation("Naive baseline (today=tomorrow)", base_rmse, base_mae, float("nan"), base_dir)
    )

    # 2) Linear Regression.
    lin_eval, _ = evaluate_model(
        "Linear Regression", LinearRegression(),
        X_train, y_train, X_test, y_test, today_close_test,
    )
    results.append(lin_eval)

    # 3) Random Forest.
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )
    rf_eval, _ = evaluate_model(
        "Random Forest", rf,
        X_train, y_train, X_test, y_test, today_close_test,
    )
    results.append(rf_eval)

    return results, (train, test)
