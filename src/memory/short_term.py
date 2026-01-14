
from typing import List, Dict

class ShortTermMemory:
    """
    Simples in-memory store for the last N conversation turns.
    Used for providing context to the Agent's planning module.
    """
    def __init__(self, limit: int = 10):
        # Format: [{"role": "user", "content": "Analyze AAPL"}, {"role": "agent", "content": "Here is the report..."}]
        self.history: List[Dict[str, str]] = []
        self.limit = limit

    def add_turn(self, user_input: str, agent_response: str):
        """Adds a turn and maintains the limit."""
        if not user_input or not agent_response:
            return

        self.history.append({"role": "user", "content": user_input})
        # Truncate agent response if it's huge (e.g., full report) to save tokens
        clean_response = agent_response[:500] + "..." if len(agent_response) > 500 else agent_response
        self.history.append({"role": "agent", "content": clean_response})
        
        # Enforce limit (limit * 2 because each turn is 2 messages)
        if len(self.history) > self.limit * 2:
            self.history = self.history[-(self.limit * 2):]

    def get_context_string(self) -> str:
        """Returns the history formatted for the LLM prompt."""
        if not self.history:
            return "No previous context."
        
        return "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in self.history])

    def clear(self):
        self.history = []

# Global Singleton
stm = ShortTermMemory(limit=10)
