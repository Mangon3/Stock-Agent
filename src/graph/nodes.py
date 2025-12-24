from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.utils.retry import retry_with_backoff

logger = setup_logger(__name__)

@retry_with_backoff(max_retries=5)
def call_model(state: AgentState, model):
    """
    Function that calls the model.
    """
    logger.info("DEBUG: Entering call_model node.")
    messages = state['messages']
    logger.info(f"DEBUG: Invoking model with {len(messages)} messages.")
    try:
        response = model.invoke(messages)
        logger.info(f"DEBUG: Model invocation successful. Output: {response.content}")
        
        if not response.tool_calls and not response.content:
            logger.warning("Empty response from model! Attempting heuristic fallback...")
            
            # Check if likely asking for micro analysis (based on last message)
            last_msg = messages[-1].content
            if "retrain" in last_msg or "analyze" in last_msg:
                 import re
                 match = re.search(r"for\s+([A-Z]+)", last_msg)
                 if match:
                     symbol = match.group(1)
                     logger.info(f"Fallback: constructing manual tool call for {symbol}")

                     from langchain_core.messages import AIMessage
                     
                     import uuid
                     call_id = str(uuid.uuid4())
                     
                     manual_call = {
                         "name": "micro_analysis",
                         "args": {"symbol": symbol},
                         "id": call_id,
                         "type": "tool_call"
                     }
                     
                     response = AIMessage(
                         content="", 
                         tool_calls=[manual_call]
                     )
        
        if response.tool_calls:
            logger.info(f"DEBUG: Tool calls: {response.tool_calls}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"DEBUG: Model invocation failed: {e}")
        raise
