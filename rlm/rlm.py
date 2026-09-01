from __future__ import annotations
from abc import ABC, abstractmethod


class RLM(ABC):
    """Abstract base class for Recursive Language Models."""

    @abstractmethod
    def completion(self, context: list[str] | str | dict[str, str], query: str) -> str:
        """
        Given a query and a (potentially long) context, return an answer.
        This is a drop-in replacement for a standard LM completion call.
        """
        pass

    @abstractmethod
    def cost_summary(self) -> dict[str, float]:
        """Return a summary of costs incurred so far."""
        pass

    @abstractmethod
    def reset(self):
        """Reset internal state (REPL environment, message history, etc.)."""
        pass
