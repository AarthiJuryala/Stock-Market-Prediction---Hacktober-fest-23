import os
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import json
from datetime import datetime
from twelvedata import TDClient 

# ---------- Configuration ----------
API_KEY = "xxxxxxxxxx"  # Replace with your key
OUTPUT_DIR = "models"
TICKERS = ['AAPL', 'AMZN', 'BLK', 'COO', 'DGX', 'ETR', 'FOX', 'GS', 'MAC', 'NFLX']
TARGETS = ['open', 'close', 'volume']
START_DATE = "2019-01-01"

# ---------- Twelve Data Client ----------
td = TDClient(apikey=API_KEY)

# ---------- Ensure models directory exists ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Training Function ----------
def train_and_save_prophet_model(ticker, target, df):
    # Prepare dataset
    df = df[['datetime', target]].dropna().rename(columns={'datetime': 'ds', target: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    # Skip if insufficient data
    if len(df) < 100:
        print(f"Skipping {ticker} {target} - not enough data")
        return

    # Train model
    model = Prophet()
    model.fit(df)

    # Save as JSON
    model_json = model_to_json(model)
    model_path = os.path.join(OUTPUT_DIR, f"{ticker}_{target}.json")
    with open(model_path, 'w') as f:
        json.dump(model_json, f)
    print(f"Saved model: {model_path}")

# ---------- Main Training Loop ----------
for ticker in TICKERS:
    print(f"\n--- Processing {ticker} ---")
    try:
        ts = td.time_series(
            symbol=ticker,
            interval="1day",
            start_date=START_DATE,
            outputsize=5000
        )
        df = ts.as_pandas()
        df = df.sort_index().reset_index().rename(columns={'datetime': 'datetime'})

        # Lowercase column names
        df.columns = [col.lower() for col in df.columns]

        for target in TARGETS:
            if target in df.columns:
                train_and_save_prophet_model(ticker, target, df)
            else:
                print(f"Column '{target}' not found for {ticker}")

    except Exception as e:
        print(f"Failed for {ticker}: {e}")
