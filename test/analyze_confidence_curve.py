
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tools.model.data import tv_data_fetcher
from src.tools.model.feature import FeatureCalculator
from src.tools.model.neural import HybridStockNet
from src.config.settings import settings
from src.utils.device import get_device

def analyze_confidence_curve(symbol="BTCUSDT"):
    device = get_device()
    print(f"Analyzing Confidence Curve for {symbol}...")
    
    # 1. Fetch & Prep (Same as before)
    # Using 730 days to match training data timeframe approx, or 300 as requested. 
    # Use settings.DATA_TIMEFRAME_DAYS for consistency or user's 300. User asked for 300 in snippet. 
    # I will use settings.DATA_TIMEFRAME_DAYS (730) to test on the full distribution the model knows or new data.
    # Actually, testing on training data is cheating. But we split inside train.py.
    # Here we are fetching NEW data potentially or overlapping.
    # Let's use the last 300 days as a validation set proxy.
    
    days = 300
    print(f"Fetching last {days} days of Hourly data...")
    df = tv_data_fetcher.fetch_historical_data(symbol, timeframe_days=days, interval="1h", exchange="BINANCE")
    
    # Fallback
    if isinstance(df, dict) and "error" in df:
         print(f"BINANCE error: {df['error']}")
         df = tv_data_fetcher.fetch_historical_data(symbol, timeframe_days=days, interval="1h", exchange="NYSE")

    if isinstance(df, dict) and "error" in df:
        print(f"Error fetching data: {df}")
        return
    
    feature_df = FeatureCalculator.calculate_features(df)
    # Join features with close price
    df = df[['close']].join(feature_df, how='inner')
    
    # Load Model
    input_dim = len(settings.FEATURE_COLUMNS)
    model = HybridStockNet(
        input_size=input_dim, 
        hidden_dim=settings.HIDDEN_DIM, 
        num_layers=settings.NUM_LAYERS, 
        dropout=settings.DROPOUT
    ).to(device)
    
    model_path = "data/datasets/models/micro.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Generate Predictions
    seq_len = settings.SEQ_LEN
    feature_vals = df[settings.FEATURE_COLUMNS].values
    
    # Vectorized Batching
    X_list = []
    actual_dir = []
    
    # We need the NEXT return to validate
    # df['target'] = log_return of t+1
    # We shift(-1) so at index i, close[i] is current, close[i+1] is next.
    # log(close[i+1]/close[i])
    # But for calculation at step i, we look at next day return.
    
    log_rets = np.log(df['close'].shift(-1) / df['close'])
    
    # We can only predict up to len(df)-2 (because last point has no target)
    # And we need seq_len history.
    
    print(f"Generating predictions for {len(df)} candles...")
    
    valid_indices = []
    
    for i in range(seq_len, len(df)-1):
        X_list.append(feature_vals[i-seq_len:i])
        # Did price go UP (1) or DOWN (0)?
        ret = log_rets.iloc[i] # This is return from i to i+1
        actual_dir.append(1 if ret > 0 else 0)
        valid_indices.append(i)
        
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Multi-Task Output: (Price, Probability)
        preds_raw, probs_raw = model(X_tensor)
        preds_raw = preds_raw.cpu().numpy().flatten()
        probs_raw = probs_raw.cpu().numpy().flatten() # We can analyze this later too
        
    # 3. Analyze Buckets
    results = pd.DataFrame({
        'Pred_Raw': preds_raw,
        'Pred_Prob': probs_raw,
        'Pred_Abs': np.abs(preds_raw),
        'Actual_Dir': actual_dir
    })
    
    # Did the model predict Up (Positive) or Down (Negative)?
    # We use the CLASSIFICATION HEAD for direction now
    results['Pred_Dir'] = (results['Pred_Prob'] > 0.5).astype(int)
    results['Is_Correct'] = (results['Pred_Dir'] == results['Actual_Dir']).astype(int)
    
    # Confidence is distance from 0.5 (Neutral)
    results['Conf_Prob'] = np.abs(results['Pred_Prob'] - 0.5)
    
    # 4. Plot Accuracy by Confidence Decile (Probability)
    try:
        results['Confidence_Bin'] = pd.qcut(results['Conf_Prob'], 10, labels=False)
    except ValueError:
        print("Not enough unique values for 10 qcut bins, trying 5...")
        results['Confidence_Bin'] = pd.qcut(results['Conf_Prob'], 5, labels=False)
    
    bin_stats = results.groupby('Confidence_Bin')['Is_Correct'].mean()
    
    print("\nAccuracy by Confidence Decile (0=Lowest Conf, 9=Highest Conf):")
    print(bin_stats)
    
    plt.figure(figsize=(10, 6))
    bin_stats.plot(kind='bar', color='skyblue')
    plt.axhline(0.5, color='red', linestyle='--', label='Random (50%)')
    plt.axhline(0.5267, color='green', linestyle='--', label='Baseline (52.7%)')
    plt.title("Does Accuracy Increase with Confidence?")
    plt.xlabel("Confidence Bin (Prediction Magnitude)")
    plt.ylabel("Directional Accuracy")
    plt.ylim(0.40, 0.70)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    out_path = "artifacts/confidence_curve.png"
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    analyze_confidence_curve()
