import logging

import httpx

from backend.config import Settings, settings
from backend.exceptions import (
    AIConfigError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)

logger = logging.getLogger(__name__)

MAX_ANSWER_CHARS = 20_000


class AIService:
    def __init__(
        self,
        config: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = config.openrouter_api_key_value
        self.model = config.openrouter_model
        self.base_url = config.openrouter_base_url
        self.timeout = config.ai_timeout_seconds
        self.max_tokens = config.ai_max_tokens
        self.temperature = config.ai_temperature
        self.transport = transport

    async def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key or not self.model:
            raise AIConfigError()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "FootyGPT",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self.timeout,
                    connect=min(self.timeout, 10.0),
                ),
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            logger.warning("OpenRouter request timed out.")
            raise AITimeoutError() from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "OpenRouter connection error: %s",
                type(exc).__name__,
            )
            raise AIProviderError() from exc

        self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            logger.warning("OpenRouter returned invalid JSON.")
            raise AIProviderError() from exc

        return self._extract_text(data)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code

        if status == 429:
            logger.warning(
                "OpenRouter rate limit: %s",
                response.text[:1000],
            )
            raise AIRateLimitError()

        if status in (401, 402, 403):
            logger.warning(
                "OpenRouter configuration error (%s): %s",
                status,
                response.text[:1000],
            )
            raise AIConfigError()

        if status >= 400:
            logger.warning(
                "OpenRouter HTTP error (%s): %s",
                status,
                response.text[:1000],
            )
            raise AIProviderError()

    @staticmethod
    def _extract_text(data: object) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("OpenRouter response did not contain expected message content.")
            raise AIProviderError() from exc

        if not isinstance(content, str) or not content.strip():
            logger.warning("OpenRouter returned an empty or unsupported answer.")
            raise AIProviderError()

        text = content.strip()

        if len(text) > MAX_ANSWER_CHARS:
            logger.warning("OpenRouter response exceeded the answer length limit.")
            raise AIProviderError()

        return text


def get_ai_service() -> AIService:
    return AIService(settings)