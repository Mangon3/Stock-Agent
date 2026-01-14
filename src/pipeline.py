from typing import Dict, Any, List
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.news import news_fetcher
from src.tools.micro import micro_model
from src.rag.core import rag_system
from src.utils.logger import setup_logger
from src.config.prompts import StockAgentPrompts

logger = setup_logger(__name__)

class StockAnalysisPipeline:
    """
    1. Macro News Analysis
    2. Micro Model Training & Inference
    3. Final Synthesis
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def run_analysis(self, symbol: str) -> Dict[str, Any]:
        logger.info(f"--- STARTING PIPELINE ANALYSIS FOR {symbol} ---")
        
        # --- Step 1: Macro News Analysis ---
        logger.info("[Step 1] Fetching Macro News Data...")
        try:
            news_data = news_fetcher.fetch_stock_news(symbol=symbol, limit=10, timeframe_days=7)
            
            rag_system.ingest_news_documents(news_data)
            query = f"Analyze the macro outlook for {symbol} based on recent news."
            context, _ = rag_system.retrieve_context(query)
            
            if not context or "No relevant" in context:
                macro_analysis_text = "No significant macro news found for this period."
            else:
                macro_analysis_text = context
                
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            macro_analysis_text = f"Error performing macro analysis: {e}"

        logger.info("[Step 1] Macro Data Ready.")

        # --- Step 2: Micro Model Analysis ---
        logger.info("[Step 2] Training & Running Micro Model...")
        try:
            micro_data = micro_model.execute_model_training(symbols_list=symbol, num_epochs=50)
        except Exception as e:
            logger.error(f"Micro analysis failed: {e}")
            micro_data = {"error": str(e), "status": "failed"}

        logger.info("[Step 2] Micro Data Ready.")

        # --- Step 3: Synthesis ---
        logger.info("[Step 3] Synthesizing Final Report...")
        final_report = self._synthesize_report(
            symbol=symbol,
            macro_text=macro_analysis_text,
            micro_data=micro_data
        )
        
        logger.info(f"--- PIPELINE COMPLETE FOR {symbol} ---")
        
        return final_report

    def _synthesize_report(self, symbol: str, macro_text: str, micro_data: Dict[str, Any]) -> str:
        
        micro_json = json.dumps(micro_data, indent=2)

        # Prepare the system prompt by formatting the template
        formatted_system_prompt = StockAgentPrompts.REPORT_SYNTHESIS_SYSTEM.format(
            symbol=symbol,
            macro_text=macro_text,
            micro_json=micro_json
        )
        
        messages = [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=StockAgentPrompts.get_report_synthesis_user_msg(symbol))
        ]
        
        return self.llm.invoke(messages).content
