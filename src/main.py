"""
main.py
=======
Entry point. Runs the full pipeline:

    load prices -> engineer features -> train/evaluate models -> report

Usage:
    python main.py                  # defaults to AAPL, 5 years
    python main.py --ticker MSFT    # any ticker
    python main.py --refresh        # force a fresh download

The script prints a comparison table and saves a results CSV plus a plot of
predicted-vs-actual closing prices to the outputs/ folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make sibling modules importable whether run from root or src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_prices
from features import build_features, feature_columns
from models import run_models, chronological_split

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Stock closing-price predictor")
    parser.add_argument("--ticker", default="AAPL", help="Stock symbol (default AAPL)")
    parser.add_argument("--period", default="5y", help="History window (default 5y)")
    parser.add_argument("--refresh", action="store_true", help="Force re-download")
    args = parser.parse_args()

    # 1) Data ---------------------------------------------------------------
    prices = load_prices(args.ticker, period=args.period, force_refresh=args.refresh)
    print(f"[main] {len(prices)} trading days loaded for {args.ticker}\n")

    # 2) Features -----------------------------------------------------------
    featured = build_features(prices)
    cols = feature_columns(featured)
    print(f"[main] Engineered {len(cols)} features: {', '.join(cols)}\n")

    # 3) Train + evaluate ---------------------------------------------------
    results, (train, test) = run_models(featured)

    # 4) Report -------------------------------------------------------------
    print("=" * 68)
    print(f"RESULTS for {args.ticker}  (train={len(train)} days, test={len(test)} days)")
    print("=" * 68)
    header = f"{'Model':<34}{'RMSE':>9}{'MAE':>9}{'Dir.Acc':>9}"
    print(header)
    print("-" * 68)
    rows = []
    for r in results:
        dir_pct = f"{r.directional_accuracy * 100:.1f}%"
        print(f"{r.name:<34}{r.rmse:>9.3f}{r.mae:>9.3f}{dir_pct:>9}")
        rows.append(
            {
                "model": r.name,
                "rmse": r.rmse,
                "mae": r.mae,
                "r2": r.r2,
                "directional_accuracy": r.directional_accuracy,
            }
        )
    print("=" * 68)

    # How much did the best real model beat the naive baseline on RMSE?
    baseline_rmse = results[0].rmse
    best_model = min(results[1:], key=lambda r: r.rmse)
    improvement = (baseline_rmse - best_model.rmse) / baseline_rmse * 100
    print(
        f"\nBest model: {best_model.name} | "
        f"RMSE improvement over naive baseline: {improvement:+.1f}%"
    )
    print(
        f"Directional accuracy: {best_model.directional_accuracy * 100:.1f}% "
        "(≈50% is expected — short-term direction is close to random)\n"
    )

    # 5) Save artifacts -----------------------------------------------------
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUTS / f"{args.ticker}_results.csv", index=False)
    print(f"[main] Saved metrics to outputs/{args.ticker}_results.csv")

    _save_plot(args.ticker, featured)


def _save_plot(ticker: str, featured: pd.DataFrame):
    """Plot actual vs predicted closing prices on the test set for the RF model."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend, no display needed
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestRegressor
        from features import feature_columns

        cols = feature_columns(featured)
        train, test = chronological_split(featured, test_frac=0.2)
        rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(train[cols], train["target_close"])
        preds = rf.predict(test[cols])

        plt.figure(figsize=(11, 5))
        plt.plot(test.index, test["target_close"].values, label="Actual close", linewidth=1.5)
        plt.plot(test.index, preds, label="Predicted close (RF)", linewidth=1.2, alpha=0.8)
        plt.title(f"{ticker} — Actual vs Predicted Next-Day Close (test set)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.tight_layout()
        out = OUTPUTS / f"{ticker}_pred_vs_actual.png"
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"[main] Saved plot to outputs/{out.name}")
    except Exception as exc:
        print(f"[main] Plot skipped ({exc})")


if __name__ == "__main__":
    main()
