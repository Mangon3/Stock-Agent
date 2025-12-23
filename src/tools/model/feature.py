import pandas as pd
import numpy as np
from typing import List, Tuple, Any

from src.config.settings import settings

class FeatureCalculator:
    
    @staticmethod
    def _calculate_true_range(df: pd.DataFrame) -> pd.Series:
        
        range1 = df['high'] - df['low']
        range2 = abs(df['high'] - df['close'].shift(1))
        range3 = abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum.reduce([range1, range2, range3])
        
        return pd.Series(true_range, index=df.index)

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        
        diff = series.diff(1)
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-8) 
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
                
        required_cols = settings.REQUIRED_OHLCV_COLS
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"DataFrame is missing required columns: {missing}")
            
        # 1. Moving Averages Ratios
        df['SMA_10_Ratio'] = df['close'].rolling(window=10).mean() / df['close']
        df['EMA_10_Ratio'] = df['close'].ewm(span=10).mean() / df['close']
        df['EMA_50_Ratio'] = df['close'].ewm(span=50).mean() / df['close']
        
        # 2. MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['MACD'] = (ema_12 - ema_26) / df['close']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        
        # 3. RSI
        df['RSI'] = FeatureCalculator._calculate_rsi(df['close'], period=14) / 100.0
        
        # 4. Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['Stochastic_K'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-8))
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

        # 5. ATR 
        df['True_Range'] = FeatureCalculator._calculate_true_range(df) 
        df['ATR_Ratio'] = df['True_Range'].ewm(span=14).mean() / df['close']

        # 6. Volume Ratio
        df['Vol_SMA_20'] = df['volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['volume'] / (df['Vol_SMA_20'] + 1e-8)
        
        # 7. OBV Slope
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['OBV_Slope'] = df['OBV'].diff(5) / (df['volume'].rolling(20).mean() * 5 + 1e-8)
       
        feature_df = df[settings.FEATURE_COLUMNS].copy()
        
        return feature_df.dropna(subset=settings.FEATURE_COLUMNS)
    
features = FeatureCalculator()