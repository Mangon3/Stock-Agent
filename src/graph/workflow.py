from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings
from src.graph.state import AgentState
from src.graph.nodes import call_model
from src.tools.registry import tools

# Initialize Model
model = ChatGoogleGenerativeAI(
    model=settings.MODEL,
    api_key=settings.GOOGLE_API_KEY,
    temperature=0.2,
    max_retries=0
)

model_with_tools = model.bind_tools(tools)

# Define Logic
def agent_node(state):
    return call_model(state, model_with_tools)

# Define Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Add Edges
workflow.set_entry_point("agent")

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()
