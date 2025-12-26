
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import settings
from src.tools.model.data import tv_data_fetcher
from src.tools.model.feature import FeatureCalculator

def run_canary_test():
    print("Starting Canary Test (Random Forest Baseline)...")
    print("-" * 60)
    
    # 1. Fetch Data
    symbol = "BTCUSDT"
    days = 730 # 2 Years
    print(f"1. Fetching HOURLY data for {symbol} ({days} days)...")
    
    from tvDatafeed import Interval
    # Try BINANCE first for Crypto
    df = tv_data_fetcher.fetch_historical_data(symbol, days, exchange="BINANCE", interval="1h")
    
    # Fallback logic similar to train.py
    if isinstance(df, dict) and "error" in df:
         print(f"   BINANCE fetch failed: {df['error']}")
         df = tv_data_fetcher.fetch_historical_data(symbol, days, exchange="NYSE", interval="1h")
         
    if isinstance(df, dict) and "error" in df:
         print(f"   NYSE fetch failed: {df['error']}")
         df = tv_data_fetcher.fetch_historical_data(symbol, days, exchange="NASDAQ", interval="1h")

    if isinstance(df, dict) and "error" in df:
         print(f"FATAL: Could not fetch data: {df['error']}")
         return
    
    # 2. Add Features
    print("2. Calculating features...")
    feature_df = FeatureCalculator.calculate_features(df)
    # Join target-relevant columns back
    df = df[['close']].join(feature_df, how='inner')
    
    # 3. Create Target
    # Target: Next day's log return * 100 (percentage)
    df['target_return'] = np.log(df['close'].shift(-1) / df['close']) * 100.0
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    df = df.dropna()
    
    print(f"3. Data Prepared. Total Samples: {len(df)}")
    
    # 4. Prepare X and y
    feature_cols = settings.FEATURE_COLUMNS
    X = df[feature_cols].values
    y_reg = df['target_return'].values
    y_cls = df['target_direction'].values
    
    # Split Data (Time-series split, no shuffle)
    # Using larger test set (20%) to match NN split roughly
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
    _, _, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, shuffle=False)
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples:  {len(X_test)}")
    
    # 5. Run Random Forest Classifier (Direction)
    print("\n[TEST A] Random Forest Classifier (Directional)")
    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_cls.fit(X_train, y_train_cls)
    
    train_acc = accuracy_score(y_train_cls, rf_cls.predict(X_train))
    test_acc = accuracy_score(y_test_cls, rf_cls.predict(X_test))
    
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy:  {test_acc:.4f}")
    print("   Test Report:")
    print(classification_report(y_test_cls, rf_cls.predict(X_test)))
    
    # 6. Run Random Forest Regressor (Magnitude)
    print("\n[TEST B] Random Forest Regressor (Returns)")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_reg.fit(X_train, y_train_reg)
    
    train_mse = mean_squared_error(y_train_reg, rf_reg.predict(X_train))
    test_mse = mean_squared_error(y_test_reg, rf_reg.predict(X_test))
    
    # Calculate R2 (Coefficient of Determination)
    train_r2 = rf_reg.score(X_train, y_train_reg)
    test_r2 = rf_reg.score(X_test, y_test_reg)
    
    print(f"   Train MSE: {train_mse:.4f}")
    print(f"   Test MSE:  {test_mse:.4f}")
    print(f"   Train R2:  {train_r2:.4f}")
    print(f"   Test R2:   {test_r2:.4f}")
    
    print("-" * 60)
    if test_acc > 0.55:
        print("CONCLUSION: Neural Net Architecture is likely the problem (RF works).")
    elif test_acc < 0.52:
        print("CONCLUSION: Features/Data are likely the problem (RF fails too).")
    else:
        print("CONCLUSION: Inconclusive (Borderline).")

if __name__ == "__main__":
    run_canary_test()
