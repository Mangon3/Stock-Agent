from typing import Optional, Dict, Any, Annotated
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from src.agent import Agent
from src.app.persistence import cache_manager
from src.utils.logger import setup_logger
from src.config.settings import settings

logger = setup_logger(__name__)


app = FastAPI(
    title="StockAgent API",
    description=settings.API_DESCRIPTION,
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

def get_rag_function_closure(agent_instance: Agent, query: str, symbol: str, timeframe: int):
    """
    Helper to create the worker function for cache invocation.
    """
    def rag_function_for_cache():
        return agent_instance.run_rag_pipeline(query, symbol)
    return rag_function_for_cache

@app.get("/")
async def root():
    return {"message": "StockAgent API is running. Use /analyze to generate reports."}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_stock(
    request: AnalyzeRequest,
    x_gemini_api_key: Annotated[str | None, Header()] = None
):
    """
    Full pipeline analysis: Macro (News/RAG) + Micro (Model) + Synthesis.
    Accepts optional X-Gemini-API-Key header.
    """
    try:
        # Determine API Key logic
        api_key = x_gemini_api_key
        if not api_key:
            api_key = settings.GOOGLE_API_KEY
            
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API Key. Provide 'X-Gemini-API-Key' header.")

        # Instantiate PER-REQUEST Agent
        current_agent = Agent(api_key=api_key)
        symbol = request.symbol.upper()
        final_report = current_agent.run_analysis(symbol)

        return AnalyzeResponse(
            symbol=symbol,
            macro_analysis="See Final Report", # Legacy field
            micro_analysis={"status": "completed"}, # Legacy field
            final_report=final_report
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

