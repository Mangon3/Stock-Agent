import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
from dotenv import load_dotenv
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

class TvDataFetcher:
    
    def __init__(self):
        self.tv = self._initialize_tv_datafeed()

    def _initialize_tv_datafeed(self):
        
        try:
            tv_instance = TvDatafeed()
            return tv_instance
        except Exception as e:
            if "driver" in str(e).lower() or "selenium" in str(e).lower():
                 logger.critical("FATAL ERROR: Failed to initialize TvDatafeed due to web driver issues.")
                 logger.critical("The library still requires a functioning Chromedriver/Selenium setup, even in anonymous mode.")
                 logger.critical(f"Original error: {e}")
            else:
                 logger.error(f"Failed to initialize TvDatafeed. {e}")
            return None

    def fetch_historical_data(self, symbol: str, timeframe_days: int, exchange: str = "NASDAQ") -> Union[pd.DataFrame, Dict[str, str]]:
        if self.tv is None:
            return {"error": "TvDatafeed is not initialized. Cannot fetch data."}

        try:
            n_bars = int(timeframe_days * 1.5)
            
            # Request data
            data = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=n_bars
            )

            if data is None or data.empty:
                return {"error": f"No historical data found for {symbol} on {exchange}."}

            data.columns = [col.lower() for col in data.columns]
            data['News_Sentiment_Score'] = 0.5 
            data = data[['open', 'high', 'low', 'close', 'volume', 'News_Sentiment_Score']]

            return data

        except Exception as e:
            return {"error": f"Data fetching error for {symbol}: {e}"}

tv_data_fetcher = TvDataFetcher()