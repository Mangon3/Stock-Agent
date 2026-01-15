
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
        # Truncate aggressively for Planning context. We just need the "gist".
        clean_response = agent_response[:200] + "..." if len(agent_response) > 200 else agent_response
        self.history.append({"role": "agent", "content": clean_response})
        
        # Enforce limit (limit * 2 because each turn is 2 messages)
        if len(self.history) > self.limit * 2:
            self.history = self.history[-(self.limit * 2):]


# Global Singleton
stm = ShortTermMemory(limit=3)  # Reduced from 10 to 3 to save tokens
