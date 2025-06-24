"""Factory for registering and creating agents."""

from typing import Dict, Type
from agents.base_agent import BaseAgent


class AgentFactory:
    """Factory for managing agent instances."""
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    _instances: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]):
        """Register an agent class."""
        cls._agents[name] = agent_class
    
    @classmethod
    def get_agent(cls, name: str, *args, **kwargs) -> BaseAgent:
        """Get or create an agent instance."""
        if name not in cls._agents:
            raise ValueError(f"Agent '{name}' not registered")
        
        # Create singleton instances
        if name not in cls._instances:
            cls._instances[name] = cls._agents[name](*args, **kwargs)
        
        return cls._instances[name]
    
    @classmethod
    def list_agents(cls) -> list:
        """List all registered agents."""
        return list(cls._agents.keys())