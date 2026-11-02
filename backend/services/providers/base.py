"""Abstract interface for AI providers."""

from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Interface every AI provider adapter must implement."""

    @property
    @abstractmethod
    def model(self) -> str | None:
        """The model identifier used by this provider."""

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the provider and return the assistant's text."""
