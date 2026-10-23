"""The only module that talks to OpenRouter.

The rest of the backend calls ``AIService.complete`` and never deals with
HTTP, headers, or API keys directly. Tests replace this class with a fake.
"""

import httpx

from backend.config import Settings, settings
from backend.exceptions import (
    AIConfigError,
    AIProviderError,
    AIRateLimitError,
    AIServiceError,
    AITimeoutError,
)


# A reply longer than this is treated as a provider error rather than stored.
MAX_ANSWER_CHARS = 20_000


class AIService:
    """Sends a chat completion request to OpenRouter and returns the reply."""

    def __init__(self, config: Settings, transport: httpx.AsyncBaseTransport | None = None):
        """Store configuration.

        ``transport`` lets tests inject a fake HTTP layer; production code
        leaves it as ``None``.
        """
        self.api_key = config.openrouter_api_key_value
        self.model = config.openrouter_model
        self.base_url = config.openrouter_base_url
        self.timeout = config.ai_timeout_seconds
        self.max_tokens = config.ai_max_tokens
        self.temperature = config.ai_temperature
        self.transport = transport

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send ``messages`` to OpenRouter and return the assistant's text.

        Raises one of the ``backend.exceptions.AIServiceError`` subclasses if
        the request cannot be completed. Error messages never include the
        API key, request headers, or provider response bodies.
        """
        if not self.api_key or not self.model:
            raise AIConfigError()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter uses these only for its own usage dashboards.
            "X-Title": "FootyGPT",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # No automatic retries: a retry after a timeout could double the cost
        # and may duplicate work. The user retries manually instead.
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10.0)),
                transport=self.transport,
            ) as client:
                response = await client.post("/chat/completions", json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise AITimeoutError() from exc
        except httpx.HTTPError as exc:
            raise AIProviderError() from exc

        self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise AIProviderError() from exc

        return self._extract_text(data)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Translate provider status codes into our error classes."""
        status = response.status_code
        if status == 429:
            raise AIRateLimitError()
        if status in (401, 402, 403):
            # Invalid key, insufficient credits, or forbidden.
            raise AIConfigError()
        if status >= 400:
            raise AIProviderError()

    @staticmethod
    def _extract_text(data: object) -> str:
        """Read ``choices[0].message.content`` and reject empty answers."""
        try:
            content = data["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIProviderError() from exc

        if not isinstance(content, str) or not content.strip():
            raise AIProviderError()

        text = content.strip()
        if len(text) > MAX_ANSWER_CHARS:
            # A pathological reply must not be stored in full or returned.
            raise AIProviderError()
        return text


def get_ai_service() -> AIService:
    """FastAPI dependency providing an AI service built from settings."""
    return AIService(settings)
