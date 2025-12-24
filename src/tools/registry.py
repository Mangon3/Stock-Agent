from typing import List, Dict, Any
from langchain_core.tools import tool
from src.tools.news import news_fetcher
from src.tools.micro import micro_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@tool
def get_recent_news(
    symbol: str,
    limit: int = 5,
    timeframe_days: int = 7
) -> List[Dict[str, Any]]:
    """
    Grab news on stock for the given symbol, limiting the number of articles 
    and the timeframe (in days).
    
    Args:
        symbol (str): The stock ticker symbol (e.g., 'MSFT').
        limit (int): Maximum number of articles to return (default: 5).
        timeframe_days (int): How many days back to search for news (default: 7).
    """
    logger.info(f"Calling Finnhub news tool for: {symbol}...")
    return news_fetcher.fetch_stock_news(symbol, limit, timeframe_days)

@tool
def micro_analysis(
    symbol: str,
    num_epochs: int = 50,
    timeframe_days: int = 150
) -> Dict[str, Any]:
    """
    Initiates the micro-model training pipeline on the specified stock symbol. 
    It fetches historical data, calculates technical features, and trains 
    a LSTM neural network model for prediction. 
    
    Use this tool when the user asks to 'train the model', 'retrain the micro-model', 
    or 'update the prediction model'.
    
    Args:
        symbol (str): The stock ticker symbol to train on (e.g., 'AAPL').
        num_epochs (int): The number of training iterations to run (default: 50).
        timeframe_days (int): The number of days of historical data to use for training (default: 150).
    """
    logger.info(f"Tool entered: micro_analysis for {symbol}")
    logger.info(f"Calling Micro-Model Training tool for: {symbol}...")
    return micro_model.execute_model_training(
        symbols_list=symbol, 
        num_epochs=num_epochs
    )

tools = [get_recent_news, micro_analysis]
