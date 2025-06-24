"""Base agent interface for all agents in the system."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Process the input and return results."""
        pass