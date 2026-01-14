import json
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config.prompts import StockAgentPrompts

from src.config.settings import settings
from src.utils.retry import retry_with_backoff
from src.pipeline import StockAnalysisPipeline
from src.utils.logger import setup_logger
from src.memory.store import memory_store

logger = setup_logger(__name__)

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
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Delegates the full analysis to the unified pipeline.
        This includes Macro (News), Micro (Model), and Synthesis.
        """
        logger.info(f"Tool Selection: StockAnalysisPipeline for {symbol}")
        return self.pipeline.run_analysis(symbol)

    @retry_with_backoff(max_retries=2)
    def parse_intent(self, query: str) -> Dict[str, Optional[str]]:
        """
        Classifies user intent and extracts symbol if present.
        Returns: {'intent': 'STOCK_QUERY'|'GENERAL_CHAT'|'UNKNOWN', 'symbol': str|None}
        """
        if not query:
            return {'intent': 'UNKNOWN', 'symbol': None}
            
        messages = [
            SystemMessage(content=StockAgentPrompts.SYMBOL_EXTRACTION_SYSTEM),
            HumanMessage(content=f"User Query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # Cleanup
        content = content.replace("'", "").replace('"', "").replace(".", "")
        
        logger.info(f"Intent Classification Raw Output: {content}")
        
        if content == "CHAT":
            return {'intent': 'GENERAL_CHAT', 'symbol': None}
        elif content == "UNKNOWN":
            return {'intent': 'UNKNOWN', 'symbol': None}
        else:
            # Assume it's a symbol
            return {'intent': 'STOCK_QUERY', 'symbol': content.upper()}

    def respond_conversational(self, query: str) -> Dict[str, Any]:
        """
        Generates a conversational response using the Agent Persona.
        """
        logger.info(f"Tool Selection: Conversational Response for query: {query}")
        
        messages = [
            SystemMessage(content=StockAgentPrompts.MAIN_AGENT_PERSONA),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        reply_text = response.content
        
        logger.info("Generated conversational response.")
        
        return {
            "symbol": "AI_AGENT",
            "final_report": reply_text
        }

    # Wrapper for legacy compatibility
    def parse_symbol(self, query: str) -> str:
        result = self.parse_intent(query)
        if result['intent'] == 'STOCK_QUERY':
            return result['symbol']
        return "UNKNOWN"
