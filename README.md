# Stock Price Predictor

A machine-learning pipeline that forecasts a stock's next-day closing price from historical data. It engineers time-series features, trains Linear Regression and Random Forest models, and evaluates them against a naive baseline using a time-aware train/test split.

**Stack:** Python · scikit-learn · pandas · NumPy · yfinance · matplotlib

---

## What it does

1. **Loads** 5 years of daily OHLCV data via yfinance, cached to CSV so runs are reproducible and offline-friendly.
2. **Engineers 15 time-series features** from past data only — lagged returns, moving averages, the price's distance from those averages, rolling volatility, momentum, and volume signals.
3. **Trains** Linear Regression and Random Forest to predict the next day's close, plus a naive "tomorrow = today" baseline.
4. **Evaluates** with RMSE, MAE, and directional accuracy on a chronological hold-out test set, and saves a metrics CSV and an actual-vs-predicted plot.

## Running it

\`\`\`bash
pip install -r requirements.txt
cd src
python main.py                 # defaults to AAPL, 5 years
python main.py --ticker MSFT   # any ticker
python main.py --refresh       # force a fresh download from Yahoo
\`\`\`

## Project structure

\`\`\`
stock-price-predictor/
├── src/
│   ├── data.py        # download + cache prices
│   ├── features.py    # engineer features, define the prediction target
│   ├── models.py      # time-aware split, train, evaluate
│   └── main.py        # runs the full pipeline and reports
├── data/              # cached CSVs (created on first run)
├── outputs/           # metrics CSV + plot
└── requirements.txt
\`\`\`

## Results (AAPL, 5 years)

Evaluated on the most recent 247 trading days (chronological hold-out test set):

| Model | RMSE | MAE | Directional Accuracy |
|---|---|---|---|
| Naive baseline (tomorrow = today) | 3.68 | 2.66 | 46.2% |
| Linear Regression | 4.11 | 3.04 | 48.2% |
| Random Forest | 23.80 | 17.64 | 45.7% |

## What I learned (the honest part)

This project is as much about **evaluating a model correctly** as building one. The real results above taught me several things worth more than a high accuracy number:

- **Directional accuracy sits near 50% (48.2%).** Predicting whether the next day closes up or down is close to a coin flip, because short-term price movement is nearly random. An honest stock project reports this rather than hiding behind an inflated figure. Beating ~50% consistently is genuinely hard, and that is the real takeaway.

- **The naive baseline is brutally strong.** Simply guessing "tomorrow's close = today's close" scored a lower RMSE (3.68) than Linear Regression (4.11). For stock prices this baseline is a tough benchmark, which is exactly why a model must be compared against it rather than against zero.

- **Random Forest underperformed badly (RMSE 23.80), and the reason is instructive.** Tree-based models cannot extrapolate beyond the price range they saw in training. When the stock moved into new price levels during the test period, the trees had no way to predict values outside their training range, so their error exploded. Linear Regression can extrapolate, which is why it did far better.

- **Avoiding look-ahead leakage is critical.** Every feature is computed from information available up to that day, and the train/test split is chronological — never shuffled — so the model is always tested on data that comes after what it trained on.

## Possible extensions

- Predict multi-day returns instead of next-day close.
- Add walk-forward (rolling-window) validation.
- Try gradient-boosted trees and compare feature importances.
- Incorporate sector or macro features beyond a single ticker's own history.
