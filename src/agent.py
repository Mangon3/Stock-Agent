from typing import List, Dict, Any
import json

from langchain_google_genai import ChatGoogleGenerativeAI



from src.config.settings import settings
from src.utils.retry import retry_with_backoff
from src.pipeline import StockAnalysisPipeline

class Agent:
    
    def __init__(self, api_key: str):
        self.GEMINI_API_KEY = api_key
        self.AGENT_MODEL = settings.MODEL
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.AGENT_MODEL,
            api_key=self.GEMINI_API_KEY,
            temperature=0.2,
            max_retries=0
        )
        self.pipeline = StockAnalysisPipeline(self.llm)

    @retry_with_backoff(max_retries=3)
    def run_analysis(self, symbol: str) -> str:
        """
        Delegates the full analysis to the unified pipeline.
        This includes Macro (News), Micro (Model), and Synthesis.
        """
        return self.pipeline.run_analysis(symbol)




