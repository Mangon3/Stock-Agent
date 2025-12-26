
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import settings
from src.tools.model.data import tv_data_fetcher
from src.tools.model.feature import FeatureCalculator
from src.tools.model.neural import HybridStockNet
from src.utils.device import get_device

# Re-implement StockDataset for standalone testing
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def create_sequences(df: pd.DataFrame, seq_len: int, feature_cols: list):
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
            
        # Target: Log returns * 100 for percentage points
        df['target'] = np.log(df['close'].shift(-1) / df['close']) * 100.0
        df = df.iloc[:-1] # Remove last row with NaN target
        
        data_matrix = df[feature_cols].values
        target_matrix = df['target'].values
        
        X, y = [], []
        for i in range(len(data_matrix) - seq_len):
            X.append(data_matrix[i:i+seq_len])
            y.append(target_matrix[i+seq_len])
            
        return np.array(X), np.array(y)

def analyze_learning():
    print("Starting Learning Analysis...")
    print(f"Settings: HIDDEN_DIM={settings.HIDDEN_DIM}, SEQ_LEN={settings.SEQ_LEN}, TIMEFRAME={settings.DATA_TIMEFRAME_DAYS} days")
    
    # 1. Fetch Data
    symbol = "BTCUSDT"
    print(f"Fetching data for {symbol}...")
    df = tv_data_fetcher.fetch_historical_data(symbol, settings.DATA_TIMEFRAME_DAYS, exchange="BINANCE")
    
    # 2. Add Features
    print("Calculating features...")
    feature_df = FeatureCalculator.calculate_features(df)
    df = df[['close']].join(feature_df, how='inner')
    
    # 3. Prepare Data
    feature_cols = settings.FEATURE_COLUMNS
    seq_len = settings.SEQ_LEN
    
    print(f"Creating sequences (Length: {seq_len})...")
    X, y = StockDataset.create_sequences(df, seq_len, feature_cols)
    print(f"Total samples: {len(X)}")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Initialize Model
    device = get_device()
    model = HybridStockNet(
        input_size=len(feature_cols),
        hidden_dim=settings.HIDDEN_DIM,
        num_layers=settings.NUM_LAYERS,
        dropout=settings.DROPOUT
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training Loop
    epochs = 50
    print("\nTraining Model...")
    print(f"{'Epoch':<5} | {'Train Loss':<12} | {'Test Loss':<12} | {'Test Acc (Direction)':<15}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_test_loss = 0
        correct_direction = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_test_loss += loss.item()
                
                # Directional Accuracy
                expected_dir = torch.sign(y_batch)
                predicted_dir = torch.sign(outputs)
                correct_direction += (expected_dir == predicted_dir).sum().item()
                total_samples += y_batch.size(0)
                
        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = (correct_direction / total_samples) * 100
        
        if (epoch + 1) % 5 == 0:
             print(f"{epoch+1:<5} | {avg_train_loss:.6f}     | {avg_test_loss:.6f}     | {test_acc:.2f}%")

if __name__ == "__main__":
    analyze_learning()
