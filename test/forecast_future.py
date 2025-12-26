import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.model.neural import HybridStockNet
from src.tools.model.feature import features
from src.config.settings import settings
from src.utils.device import get_device
from src.tools.model.data import TvDataFetcher

def forecast_future(symbol="AAPL", forecast_days=30):
    device = get_device()
    print(f"Device: {device}")

    # 1. Load Data
    tv_data_fetcher = TvDataFetcher()
    print(f"Fetching data for {symbol}...")
    df = tv_data_fetcher.fetch_historical_data(symbol, timeframe_days=365*2, exchange="NASDAQ")
    if isinstance(df, dict) and "error" in df:
        print("Error fetching data")
        return

    # 2. Load Model
    # We need to instantiate the model with correct dimensions
    # First, get features to know input size
    try:
        feat_df = features.calculate_features(df.copy())
        input_dim = len(settings.FEATURE_COLUMNS)
    except Exception as e:
        print(f"Feature calc error: {e}")
        return

    model = HybridStockNet(
        input_size=input_dim, 
        hidden_dim=settings.HIDDEN_DIM, 
        num_layers=settings.NUM_LAYERS, 
        dropout=settings.DROPOUT
    ).to(device)

    
    model_path = "data/datasets/models/micro.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded.")
    else:
        print("Model not found!")
        return
        
    model.eval()
    future_prices = []
    original_len = len(df)
    
    print(f"Starting {forecast_days} day forecast loop...")
    
    current_df = df.copy()
    
    SEQ_LEN = settings.SEQ_LEN  # e.g. 60
    
    with torch.no_grad():
        for i in range(forecast_days):
            # A. Calculate features on current extended stats
            # Note: This is inefficient (re-calc all), but robust for correctness of lagging indicators
            feat_df = features.calculate_features(current_df.copy())
            
            # B. Get the last sequence
            latest_features = feat_df[settings.FEATURE_COLUMNS].values[-SEQ_LEN:]
            
            if len(latest_features) < SEQ_LEN:
                print("Not enough data for sequence.")
                break
                
            tensor_seq = torch.tensor(latest_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # C. Predict
            # Multi-Task Output: (Price, Probability)
            pred_log_ret_scaled, _ = model(tensor_seq) 
            pred_log_ret_scaled = pred_log_ret_scaled.item() 
            pred_log_ret = pred_log_ret_scaled / 100.0
            
            # D. Calculate Next Price
            last_price = current_df['close'].iloc[-1]
            next_price = last_price * np.exp(pred_log_ret)
            
            future_prices.append(next_price)
            
            # E. Append to DataFrame
            next_date = current_df.index[-1] + pd.Timedelta(days=1)
            
            # Simulate Volatility for Features (High/Low)
            # If we just set High=Low=Close, volatility features (ATR, StdDev) collapse to 0,
            # and the model learns to predict 0 volatility.
            # We use the recent volatility to project realistic High/Low.
            recent_vol = current_df['close'].pct_change().rolling(14).std().iloc[-1]
            if np.isnan(recent_vol): recent_vol = 0.01  # Default fallback
            
            # Randomize slightly for realistic simulation
            sim_high = next_price * (1 + abs(np.random.normal(recent_vol, recent_vol*0.1)))
            sim_low = next_price * (1 - abs(np.random.normal(recent_vol, recent_vol*0.1)))
            
            avg_vol = current_df['volume'].iloc[-5:].mean()
            
            # Propagate Sentiment (Random Walk)
            last_sentiment = current_df['News_Sentiment_Score'].iloc[-1]
            next_sentiment = np.clip(last_sentiment + np.random.normal(0, 0.05), 0, 1) # Small drift
            
            new_row = pd.Series({
                'open': next_price,
                'high': sim_high,
                'low': sim_low,
                'close': next_price,
                'volume': avg_vol,
                'News_Sentiment_Score': next_sentiment
            }, name=next_date)
            
            current_df = pd.concat([current_df, new_row.to_frame().T])
            
            print(f"Day +{i+1}: Price={next_price:.2f} (LogRet={pred_log_ret:.5f}) | Sent={next_sentiment:.2f}")

    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot last 100 days of ACTUAL
    plt.plot(df.index[-100:], df['close'].iloc[-100:], label='Historical Data', color='blue')
    
    # Plot Future
    # Generate date range for future
    future_dates = current_df.index[original_len:]
    plt.plot(future_dates, future_prices, label='Autoregressive Forecast (Next 30 Days)', color='red', linestyle='dashed', marker='o', markersize=3)
    
    plt.title(f"Autoregressive Future Forecast: {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "artifacts/future_forecast.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    forecast_future()
