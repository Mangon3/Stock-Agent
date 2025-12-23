import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent import agent
from src.app.persistence import cache_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


app = FastAPI(
    title="StockAgent API",
    description="API for StockAgent - Financial Analysis and Prediction",
    version="1.0.0"
)


agent_executor, llm = agent.exec_agent()

class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe_days: Optional[int] = 7
    query: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "MSFT",
                "timeframe_days": 7,
                "query": "Analyze the macro outlook focusing on AI developments."
            }
        }

class AnalyzeResponse(BaseModel):
    symbol: str
    macro_analysis: str
    micro_analysis: Dict[str, Any]
    final_report: str

def get_rag_function_closure(query: str, symbol: str, timeframe: int):
    """
    Helper to create the worker function for cache invocation.
    """
    def rag_function_for_cache():
        return agent.run_rag_pipeline(query, symbol)
    return rag_function_for_cache

@app.get("/")
async def root():
    return {"message": "StockAgent API is running. Use /analyze to generate reports."}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_stock(request: AnalyzeRequest):
    """
    Full pipeline analysis: Macro (News/RAG) + Micro (Model) + Synthesis.
    """
    try:
        symbol = request.symbol.upper()
        timeframe = request.timeframe_days
        
        # Default query if none provided
        rag_query = request.query
        if not rag_query:
            rag_query = f"Analyze the macro outlook for {symbol} based on the last {timeframe} days of news."

        # === Step 1: Macro Analysis (RAG) ===
        valid_rag_function = get_rag_function_closure(rag_query, symbol, timeframe)
        
        macro_response_dict = cache_manager.invoke_with_cache(
            worker_function=valid_rag_function,
            input_query=rag_query,
            symbol=symbol,
            timeframe_days=timeframe
        )
        macro_analysis_text = macro_response_dict["output"]

        # === Step 2: Micro Analysis (Model) ===
        micro_query = f"Please retrain the prediction model for {symbol} using the default parameters."
        
        # Invoke LangGraph
        from langchain_core.messages import HumanMessage
        graph_input = {"messages": [HumanMessage(content=micro_query)]}
        
        logger.info("DEBUG: About to invoke LangGraph executor.")
        try:
            micro_response = agent_executor.invoke(graph_input)
            if "messages" not in micro_response:
                 logger.error(f"Unexpected response format from graph: {micro_response.keys()}")
                 raise ValueError("Graph did not return messages.")
        except Exception as e:
            logger.error(f"DEBUG: LangGraph executor failed: {e}")
            raise
        
        last_message = micro_response["messages"][-1]
        micro_output = last_message.content
        
        if isinstance(micro_output, list):
            text_parts = [block.get('text', '') for block in micro_output if isinstance(block, dict) and block.get('type') == 'text']
            micro_output = "\n".join(text_parts)
            if not micro_output:
                micro_output = str(last_message.content)
        
        try:
            if isinstance(micro_output, str):
                try:
                    micro_analysis_data = json.loads(micro_output)
                except json.JSONDecodeError:
                    micro_analysis_data = {"result_summary": micro_output}
            else:
                micro_analysis_data = {"raw_output": str(micro_output)}

        except Exception as e:
            logger.error(f"Error parsing micro analysis: {e}")

            micro_analysis_data = {"error": "Failed to parse micro model output", "details": str(e)}

        # === Step 3: Synthesis ===
        final_report = agent.synthesize_final_report(
            original_query=rag_query,
            macro_analysis=macro_analysis_text,
            micro_analysis_data=micro_analysis_data
        )

        return AnalyzeResponse(
            symbol=symbol,
            macro_analysis=macro_analysis_text,
            micro_analysis=micro_analysis_data,
            final_report=final_report
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
