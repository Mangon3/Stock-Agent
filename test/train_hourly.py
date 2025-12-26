
import sys
import os
import logging
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tools.model.train import trainer
from src.config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def run_hourly_training():
    print("Starting Hourly Training check...")
    print(f"Settings: Interval={settings.DATA_INTERVAL}, Timeframe={settings.DATA_TIMEFRAME_DAYS} days")
    print(f"Features: {settings.FEATURE_COLUMNS}")
    
    # Symbols to train on (Focus on BTC first)
    symbols = ['BTCUSDT', 'AAPL']
    
    # Run training
    # Increased batch size for hourly data stability
    result = trainer.train(
        symbols=symbols,
        num_epochs=30,  # 30 epochs should be enough
        batch_size=64   # Higher batch size for larger dataset
    )
    
    print("\nTraining Result:")
    print(result)

if __name__ == "__main__":
    run_hourly_training()
