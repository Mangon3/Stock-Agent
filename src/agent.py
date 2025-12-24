from typing import List, Dict, Any
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import HumanMessage


from src.config.settings import settings
from src.graph.workflow import create_workflow
from src.utils.retry import retry_with_backoff
from src.tools.registry import get_recent_news
from src.rag.core import rag_system

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
        
        # Create per-user graph
        self.graph = create_workflow(self.llm)
        self.agent_executor = self.graph 

    def get_executor(self):
        return self.agent_executor


    @retry_with_backoff(max_retries=5)
    def call_llm_with_retry(self, prompt):
        return self.llm.invoke(prompt)

    # Removed blanket retry decorator to prevent re-running news/embeddings on LLM rate limit
    def run_rag_pipeline(
        self,
        query,
        symbol
    ) -> str:
        
        global TEST_TIMEFRAME
        
        try:
            timeframe = TEST_TIMEFRAME 
        except NameError:
             timeframe = 7
            
        try:
            self.raw_news_data = get_recent_news.func(
                symbol=symbol,
                timeframe_days=timeframe
            )
            if isinstance(self.raw_news_data, list) and len(self.raw_news_data) > 0:
                 if "error" in self.raw_news_data[0]:
                     return f"Error from news tool: {self.raw_news_data[0]['error']}"
                 if "warning" in self.raw_news_data[0]:
                     pass
            elif isinstance(self.raw_news_data, str) and self.raw_news_data.startswith("Error"):
                 return self.raw_news_data
        except Exception as e:
            return f"Error: News fetching tool failed: {e}"
            
        if not isinstance(self.raw_news_data, list) or not self.raw_news_data:
            return f"Error: News fetching tool returned empty or invalid data structure for symbol {symbol}."
        
        
        rag_system.ingest_news_documents(self.raw_news_data)

        self.context, self.sources = rag_system.retrieve_context(query)

        self.final_system_prompt = (
            "You are a world-class Macro Financial News Analyst AI. Your response MUST be "
            "strictly based on the 'CONTEXT' provided below. Do not use external knowledge or invent facts. "
            "Cite the original source of the information (e.g., '[Source: Reuters]').\n\n"
            "--- CONTEXT ---\n"
            f"{self.context}\n"
            "---------------\n"
        )
        
        self.final_prompt = [
            SystemMessage(content=self.final_system_prompt),
            HumanMessage(content=query)
        ]
        
        # Use the internal retry method for just the LLM call
        response_msg = self.call_llm_with_retry(self.final_prompt)
        final_response = response_msg.content
        
        return final_response
    
    @retry_with_backoff(max_retries=5)
    def synthesize_final_report(self, original_query: str, macro_analysis: str, micro_analysis_data: Dict[str, Any]) -> str:
        """
        Runs a final LLM call to synthesize the macro (news) and micro (model) results 
        into a single, coherent investment report.
        """
        
        micro_analysis_json_str = json.dumps(micro_analysis_data, indent=2)
        
        synthesis_prompt = (
            "You are a Senior Investment Analyst. Your task is to combine the results from a Macro News Analysis "
            "and a Micro Prediction Model into a single, cohesive, and actionable investment report. "
            "Follow the thought process outlined below to generate the FINAL REPORT."
            "\n\n"
            "*** THOUGHT PROCESS ***\n\n"
            "1. **Macro Analysis (Sentiment):** Summarize the key drivers and risks identified in the Macro News Analysis. Determine the overall sentiment (Bullish/Bearish/Neutral) based on this news context."
            "\n"
            "2. **Micro Analysis (Technical):** Extract the following key metrics from the Micro Model Data: Latest Close Price, Model Signal, Confidence Level. Summarize what the model is predicting."
            "\n"
            "3. **Synthesis & Conclusion:** Compare the Macro Sentiment (from news) with the Micro Signal (from model). Are they aligned, or are they contradictory? State the final, combined investment thesis and outlook for the stock."
            "\n\n"
            "*** INPUT DATA ***\n"
            f"ORIGINAL USER QUERY: {original_query}\n\n"
            "--- MACRO NEWS ANALYSIS (Qualitative) ---\n"
            f"{macro_analysis}\n\n"
            "--- MICRO MODEL DATA (Quantitative) ---\n"
            f"{micro_analysis_json_str}\n\n"
            "*** FINAL REPORT ***\n"
        )
        
        self.final_prompt = [
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content="Generate the comprehensive investment report based on the combined analysis.")
        ]
        
        final_response = self.llm.invoke(self.final_prompt).content
        return final_response


if __name__ == "__main__": # TEST FUNCTION
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