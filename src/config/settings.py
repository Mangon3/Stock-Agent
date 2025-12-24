from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path

class Settings(BaseSettings):

    # Google / Gemini
    GOOGLE_API_KEY: Optional[str] = None
    MODEL: str = "gemini-2.5-flash"
    
    # Embedding
    TEXT_EMBEDDING: str = "text-embedding-004"
    
    # Redis
    REDIS_URL: Optional[str] = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Finnhub
    FINNHUB_API_KEY: Optional[str] = None

    # Model Parameters
    SEQ_LEN: int = 30
    HIDDEN_DIM: int = 64
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    
    # Feature Configuration
    FEATURE_COLUMNS: List[str] = [
        'SMA_10_Ratio', 'EMA_10_Ratio', 'EMA_50_Ratio', 
        'MACD', 'Signal_Line', 'RSI', 
        'Stochastic_K', 'Stochastic_D',
        'ATR_Ratio', 'Vol_Ratio', 'OBV_Slope', 
        'News_Sentiment_Score' 
    ]

    # Data Configuration
    REQUIRED_OHLCV_COLS: List[str] = ['open', 'high', 'low', 'close', 'volume', 'News_Sentiment_Score']
    DATA_TIMEFRAME_DAYS: int = 365
    
    @property
    def INPUT_SIZE(self) -> int:
        return len(self.FEATURE_COLUMNS)

    @property
    def PROJECT_ROOT(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent
        
    @property
    def MODEL_PATH(self) -> Path:
        if Path("/data").exists():
            return Path("/data/datasets/models/micro.pth")
        return self.PROJECT_ROOT / "data" / "datasets" / "models" / "micro.pth"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
