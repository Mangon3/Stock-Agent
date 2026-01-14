from typing import Optional, Dict, Any, Annotated
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, model_validator
from fastapi.middleware.cors import CORSMiddleware

from src.agent import Agent
from src.app.persistence import cache_manager
from src.utils.logger import setup_logger
from src.config.settings import settings
from src.memory.store import memory_store

logger = setup_logger(__name__)


app = FastAPI(
    title="StockAgent API",
    description=settings.API_DESCRIPTION,
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe_days: Optional[int] = 7
    query: Optional[str] = None
    
    @model_validator(mode='after')
    def check_symbol_or_query(self) -> 'AnalyzeRequest':
        if not self.symbol and not self.query:
            raise ValueError('Either symbol or query must be provided.')
        return self

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
    macro_analysis: str = "N/A"
    micro_analysis: Dict[str, Any] = {}
    final_report: str

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
        
        # Determine Intent
        intent = "STOCK_QUERY"
        symbol = request.symbol
        query_text = request.query or (f"Analyze {symbol}" if symbol else "")
        
        if not symbol and request.query:
            logger.info(f"Analyzing intent for query: {request.query}")
            intent_data = current_agent.parse_intent(request.query)
            intent = intent_data['intent']
            symbol = intent_data['symbol']
            
            logger.info(f"Determined Intent: {intent} | Symbol: {symbol}")
            
            if intent == "UNKNOWN":
                raise HTTPException(status_code=422, detail="Could not understand the query. Please identify a stock or ask a question.")
        
        result_data = None

        # Execute based on Intent
        if intent == "STOCK_QUERY" and symbol:
            symbol = symbol.upper()
            result_data = current_agent.analyze(symbol)
            # Ensure symbol is in result
            result_data['symbol'] = symbol
            
        elif intent == "GENERAL_CHAT":
            result_data = current_agent.respond_conversational(request.query)
            
        else:
             # Fallback
             raise HTTPException(status_code=422, detail="Invalid Request State.")

        # Save to Memory
        if result_data and 'final_report' in result_data:
            memory_store.save_turn(
                user_input=query_text,
                model_output=result_data['final_report'],
                intent=intent
            )

        return AnalyzeResponse(
            symbol=result_data.get('symbol', 'UNKNOWN'),
            macro_analysis=result_data.get('macro_analysis', 'See Final Report'),
            micro_analysis=result_data.get('micro_analysis', {}),
            final_report=result_data.get('final_report', "No report generated.")
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
