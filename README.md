# Stock Price Predictor

A machine-learning pipeline that forecasts a stock's next-day closing price from
historical data. It engineers time-series features, trains Linear Regression and
Random Forest models, and evaluates them against a naive baseline using a
time-aware train/test split.

**Stack:** Python · scikit-learn · pandas · NumPy · yfinance · matplotlib

---

## What it does

1. **Loads** 5 years of daily OHLCV data via yfinance, cached to CSV so runs are
   reproducible and offline-friendly.
2. **Engineers 15 time-series features** from past data only — lagged returns,
   moving averages, the price's distance from those averages, rolling
   volatility, momentum, and volume signals.
3. **Trains** Linear Regression and Random Forest to predict the next day's
   close, plus a naive "tomorrow = today" baseline.
4. **Evaluates** with RMSE, MAE, and directional accuracy on a chronological
   hold-out test set, and saves a metrics CSV and an actual-vs-predicted plot.

## Running it

```bash
pip install -r requirements.txt
cd src
python main.py                 # defaults to AAPL, 5 years
python main.py --ticker MSFT   # any ticker
python main.py --refresh       # force a fresh download from Yahoo
```

## Project structure

```
stock-price-predictor/
├── src/
│   ├── data.py        # download + cache prices
│   ├── features.py    # engineer features, define the prediction target
│   ├── models.py      # time-aware split, train, evaluate
│   └── main.py        # runs the full pipeline and reports
├── data/              # cached CSVs (created on first run)
├── outputs/           # metrics CSV + plot
└── requirements.txt
```

## What I learned (the honest part)

This project is as much about **evaluating a model correctly** as building one.
A few things the numbers taught me:

- **Predicting the price *level* is easy; predicting *direction* is hard.** The
  actual-vs-predicted plot looks impressively tight, but that is mostly because
  tomorrow's price is usually close to today's. The metric that actually matters
  for any trading decision is **directional accuracy** — did the model correctly
  call up vs down — and that lands near **50%**, because short-term price
  movement is close to random. An honest stock project reports this rather than
  hiding it.

- **The naive baseline is brutally strong.** Simply guessing "tomorrow's close =
  today's close" is a hard benchmark to beat on RMSE. Comparing against it (not
  against zero) is the only fair way to judge a model.

- **Avoiding look-ahead leakage is critical.** Every feature is computed from
  information available up to that day, and the train/test split is
  chronological — never shuffled — so the model is always tested on data that
  comes *after* what it trained on. Shuffling time-series data is the most
  common way these projects accidentally inflate their own results.

## Possible extensions

- Predict multi-day returns instead of next-day close.
- Add walk-forward (rolling-window) validation.
- Try gradient-boosted trees and compare feature importances.
- Incorporate sector or macro features beyond a single ticker's own history.
