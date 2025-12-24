from typing import List, Dict, Any
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import HumanMessage


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





# TEST FUNCTION


if __name__ == "__main__": 
    pass
    global TEST_TIMEFRAME
    
    agent_executor, llm = agent.exec_agent()
    
    TEST_SYMBOL = "MSFT"
    TEST_TIMEFRAME = 14
    RAG_USER_QUERY = f"Analyze the macro outlook for {TEST_SYMBOL} based on the last {TEST_TIMEFRAME} days of news, focusing on AI developments."
    
    def rag_function_for_cache():
        return agent.run_rag_pipeline(RAG_USER_QUERY, TEST_SYMBOL)
    
    logger.info("="*80)
    logger.info(f"STEP 1: Executing Macro News Analysis (RAG)")
    logger.info("="*80)
    
    macro_response_dict = cache_manager.invoke_with_cache(
        worker_function=rag_function_for_cache, 
        input_query=RAG_USER_QUERY, 
        symbol=TEST_SYMBOL, 
        timeframe_days=TEST_TIMEFRAME
    )
    macro_analysis_text = macro_response_dict["output"]
    
    logger.info("[MACRO ANALYSIS COLLECTED]")
    logger.info(macro_analysis_text)
    
    
    # === Step 2: Run Micro Model Training and Inference (Tool: micro_analysis) ===
    MICRO_USER_QUERY = f"Please retrain the prediction model for {TEST_SYMBOL} using the default parameters."
    
    logger.info("="*80)
    logger.info(f"STEP 2: Executing Micro Model Training and Inference (Tool: micro_analysis)")
    logger.info("="*80)

    
    graph_input = {"messages": [HumanMessage(content=MICRO_USER_QUERY)]}
    graph_response = agent_executor.invoke(graph_input)
    
    try:
        micro_output_content = graph_response["messages"][-1].content
        if isinstance(micro_output_content, str):
            try:
                micro_analysis_data = json.loads(micro_output_content)
            except json.JSONDecodeError:
                micro_analysis_data = {"raw_output": micro_output_content}
        else:
             micro_analysis_data = micro_output_content
    except Exception as e:
        logger.warning(f"Failed to parse micro model output: {e}")
        micro_analysis_data = {"raw_output": str(graph_response)}

        
    logger.info("[MICRO ANALYSIS DATA COLLECTED]")
    logger.info(json.dumps(micro_analysis_data, indent=2))
    
    
    # === Step 3: Synthesize Final Report ===
    logger.info(f"STEP 3: Synthesizing Final Combined Report...")
    
    final_report = agent.synthesize_final_report(
        original_query=RAG_USER_QUERY,
        macro_analysis=macro_analysis_text,
        micro_analysis_data=micro_analysis_data
    )
    
    logger.info("FINAL INTEGRATED INVESTMENT REPORT:")
    logger.info(final_report)