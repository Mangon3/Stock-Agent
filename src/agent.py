import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config.prompts import StockAgentPrompts

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

    @retry_with_backoff(max_retries=2)
    def parse_symbol(self, query: str) -> str:
        """
        Uses the LLM to extract a stock symbol from a natural language query.
        Returns 'UNKNOWN' if no symbol is found.
        """
        if not query:
            return "UNKNOWN"
            
        messages = [
            SystemMessage(content=StockAgentPrompts.SYMBOL_EXTRACTION_SYSTEM),
            HumanMessage(content=f"User Query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        symbol = response.content.strip().upper()
        
        # Basic cleanup in case of extra chars
        symbol = symbol.replace("'", "").replace('"', "").replace(".", "")
        
        return symbol
