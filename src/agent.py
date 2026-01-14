import json
from typing import List, Dict, Any, Optional, Generator
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

    # Note: Retries on a generator need care, but if the generator initialization fails, this helps.
    # Once yielding starts, retry won't verify mid-stream errors easily.
    @retry_with_backoff(max_retries=3)
    def analyze(self, symbol: str, tools: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Delegates to pipeline stream with selected tools.
        """
        logger.info(f"Tool Selection: StockAnalysisPipeline for {symbol} with tools {tools}")
        yield from self.pipeline.run_analysis(symbol, tools)

    @retry_with_backoff(max_retries=2)
    def parse_intent(self, query: str) -> Dict[str, Any]:
        """
        Classifies user intent and plans tool usage.
        Returns: { 'intent': str, 'symbol': str|None, 'tools': List[str] }
        """
        if not query:
            return {'intent': 'UNKNOWN', 'symbol': None, 'tools': []}
            
        messages = [
            SystemMessage(content=StockAgentPrompts.PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # Cleanup JSON if LLM wraps in backticks
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        logger.info(f"Intent Planning Raw Output: {content}")
        
        try:
            plan = json.loads(content)
            # Normalize defaults
            if "tools" not in plan:
                plan["tools"] = ["macro", "micro"]
            if "symbol" in plan and plan["symbol"]:
                plan["symbol"] = plan["symbol"].upper()
                
            return plan
        except json.JSONDecodeError:
            logger.error("Failed to parse planning JSON. Fallback to Unknown.")
            return {'intent': 'UNKNOWN', 'symbol': None, 'tools': []}

    def respond_conversational(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Generates a conversational response. Yields progress then result.
        """
        logger.info(f"Tool Selection: Conversational Response for query: {query}")
        
        yield {"type": "progress", "step": "think", "message": "Thinking...", "percent": 20}
        
        messages = [
            SystemMessage(content=StockAgentPrompts.MAIN_AGENT_PERSONA),
            HumanMessage(content=query)
        ]
        
        yield {"type": "progress", "step": "generate", "message": "Typing response...", "percent": 60}
        
        response = self.llm.invoke(messages)
        reply_text = response.content
        
        logger.info("Generated conversational response.")
        
        yield {"type": "progress", "step": "complete", "message": "Done.", "percent": 100}
        
        yield {
            "type": "result",
            "symbol": "AI_AGENT",
            "final_report": reply_text
        }

    # Wrapper for legacy compatibility (synchronous fallbacks if needed, but unused by new API)
    def parse_symbol(self, query: str) -> str:
        result = self.parse_intent(query)
        if result['intent'] == 'STOCK_QUERY':
            return result['symbol']
        return "UNKNOWN"
