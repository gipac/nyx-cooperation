"""
Base Backend Interface for LLM Integration
All LLM backends must implement this interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    text: str
    model: str
    tokens_used: int
    response_time: float
    temperature: float
    metadata: Dict[str, Any]


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends

    Supports: Ollama, HuggingFace, OpenAI-compatible APIs
    """

    def __init__(self,
                 model_name: str,
                 temperature: float = 0.7,
                 max_tokens: int = 150,
                 timeout: int = 30):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0

    @abstractmethod
    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate response from LLM

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Backend-specific parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and responsive"""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0

        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'total_time_seconds': self.total_time,
            'average_response_time': avg_time,
            'model': self.model_name,
            'temperature': self.temperature
        }

    def reset_statistics(self):
        """Reset usage statistics"""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0

    def _update_statistics(self, response: LLMResponse):
        """Update internal statistics"""
        self.total_calls += 1
        self.total_tokens += response.tokens_used
        self.total_time += response.response_time

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, temp={self.temperature})"

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(model='{self.model_name}', "
                f"temperature={self.temperature}, max_tokens={self.max_tokens})")
