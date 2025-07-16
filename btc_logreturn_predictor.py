import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import time
import os

# ===== CONFIGURATION =====
COINGECKO_DAYS = 'max'  # all available history
TEST_DAYS = 60          # last N days as test set
CRYPTOPANIC_API_KEY = 'YOUR_CRYPTOPANIC_API_KEY'  # <-- Register at cryptopanic.com for free API key
SYMBOL = 'bitcoin'
VS_CURRENCY = 'usd'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 1. FETCH DATA FROM COINGECKO =====
def fetch_coingecko_ohlcv(symbol=SYMBOL, vs_currency=VS_CURRENCY, days=COINGECKO_DAYS):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    resp = requests.get(url, params=params)
    data = resp.json()
    prices = data['prices']
    volumes = data['total_volumes']
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["volume"] = [v[1] for v in volumes]
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Create fake O/H/L using close + shift (CoinGecko daily = close only)
    df["open"] = df["close"].shift(1)
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df = df.dropna().reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df

# ===== 2. TECHNICAL INDICATORS =====
def add_indicators(df):
    df['macd'] = MACD(close=df['close']).macd()
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    bb = BollingerBands(close=df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    return df

# ===== 3. NEWS SENTIMENT FROM CRYPTOPANIC =====
def fetch_news_sentiment(start_date, end_date, api_key=CRYPTOPANIC_API_KEY):
    # One request per day to avoid API limit
    day = start_date
    news_by_date = {}
    headers = {"Accept": "application/json"}
    while day <= end_date:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies=BTC&filter=all&public=true"
        params = {"published_after": day.strftime('%Y-%m-%dT00:00:00'), "published_before": (day + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')}
        try:
            resp = requests.get(url, params=params, headers=headers)
            news = resp.json().get('results', [])
        except Exception as e:
            print(f"Error fetching news for {day.date()}: {e}")
            news = []
        # Score: +1 (positive), 0 (neutral), -1 (negative)
        pos = sum(1 for n in news if n.get("votes", {}).get("positive", 0) > n.get("votes", {}).get("negative", 0))
        neg = sum(1 for n in news if n.get("votes", {}).get("negative", 0) > n.get("votes", {}).get("positive", 0))
        neu = len(news) - pos - neg
        total = len(news) if len(news) > 0 else 1
        news_by_date[day.date()] = (pos - neg) / total
        time.sleep(1)  # Be gentle with API
        day += timedelta(days=1)
    return news_by_date

# ===== 4. PREPARE DATASET =====
def prepare_dataset():
    df = fetch_coingecko_ohlcv()
    df = add_indicators(df)
    # Calculate 1-day log-return
    df['future_close'] = df['close'].shift(-1)
    df['log_return'] = np.log(df['future_close'] / df['close'])
    df = df.dropna().reset_index(drop=True)
    # Add news sentiment
    start = df['timestamp'].min().date()
    end = df['timestamp'].max().date()
    print("Fetching news sentiment. This may take a few minutes...")
    news_sentiment = fetch_news_sentiment(start, end)
    df['news_sentiment'] = df['timestamp'].dt.date.map(news_sentiment)
    return df

# ===== 5. SPLIT TRAIN/TEST =====
def split_data(df, test_days=60):
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test

# ===== 6. TRAIN & PREDICT LIGHTGBM =====
def train_and_predict(train, test):
    FEATURES = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'news_sentiment']
    TARGET = 'log_return'
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'seed': 42
    }
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=20, verbose_eval=20)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred, X_test

# ===== 7. EVALUATE & SAVE =====
def evaluate_and_save(model, y_test, y_pred, X_test):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nTest MAE: {mae:.6f} | Test RMSE: {rmse:.6f}")
    # Save predictions
    results = pd.DataFrame(X_test).copy()
    results['true_log_return'] = y_test.values
    results['pred_log_return'] = y_pred
    results.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    # Save model
    joblib.dump(model, f"{OUTPUT_DIR}/lgbm_model.pkl")
    print(f"\nResults and model saved in ./{OUTPUT_DIR}/")

# ===== MAIN RUN =====
if __name__ == "__main__":
    df = prepare_dataset()
    train, test = split_data(df)
    model, y_test, y_pred, X_test = train_and_predict(train, test)
    evaluate_and_save(model, y_test, y_pred, X_test)