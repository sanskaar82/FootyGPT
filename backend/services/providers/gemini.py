"""Google Gemini provider adapter."""

import logging

import httpx

from backend.config import Settings
from backend.exceptions import (
    AIConfigError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from backend.services.providers.base import AIProvider

logger = logging.getLogger(__name__)

MAX_ANSWER_CHARS = 20_000


class GeminiProvider(AIProvider):
    """Google Gemini generateContent adapter.

    Sends prompts to Gemini's ``/models/{model}:generateContent`` endpoint.
    Authentication uses the ``x-goog-api-key`` header. API keys never appear
    in logs or raised exceptions.
    """

    def __init__(
        self,
        config: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = config.gemini_api_key_value
        self._model = config.gemini_model
        self.base_url = config.gemini_base_url
        self.timeout = config.ai_timeout_seconds
        self.max_tokens = config.ai_max_tokens
        self.temperature = config.ai_temperature
        self.transport = transport

    @property
    def model(self) -> str | None:
        return self._model

    async def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key or not self._model:
            raise AIConfigError()

        system_instruction, contents = self._map_messages(messages)

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        url = f"/models/{self._model}:generateContent"

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
                    url,
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            logger.warning("Gemini request timed out.")
            raise AITimeoutError() from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "Gemini connection error: %s",
                type(exc).__name__,
            )
            raise AIProviderError() from exc

        self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            logger.warning("Gemini returned invalid JSON.")
            raise AIProviderError() from exc

        return self._extract_text(data)

    @staticmethod
    def _map_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, object]]]:
        """Map OpenAI-style messages to Gemini systemInstruction + contents.

        System messages are combined into ``systemInstruction`` rather than
        being placed in the contents array. User messages map to ``user``
        role; assistant messages map to ``model`` role.
        """
        system_parts: list[str] = []
        contents: list[dict[str, object]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                system_parts.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"

            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": content}],
                }
            )

        system_instruction = "\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code

        if status == 429:
            logger.warning(
                "Gemini rate limit: %s",
                response.text[:1000],
            )
            raise AIRateLimitError()

        if status in (401, 403):
            logger.warning(
                "Gemini configuration error (%s): %s",
                status,
                response.text[:1000],
            )
            raise AIConfigError()

        if status >= 500:
            logger.warning(
                "Gemini server error (%s): %s",
                status,
                response.text[:1000],
            )
            raise AIProviderError(retryable=True) from None

        if status >= 400:
            logger.warning(
                "Gemini HTTP error (%s): %s",
                status,
                response.text[:1000],
            )
            raise AIProviderError() from None

    @staticmethod
    def _extract_text(data: object) -> str:
        try:
            candidates = data["candidates"]
        except (KeyError, TypeError) as exc:
            logger.warning("Gemini response did not contain candidates.")
            raise AIProviderError() from exc

        if not isinstance(candidates, list) or len(candidates) == 0:
            logger.warning("Gemini returned no candidates.")
            raise AIProviderError()

        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            logger.warning("Gemini candidate is not an object.")
            raise AIProviderError()

        content = first_candidate.get("content")
        if not isinstance(content, dict):
            logger.warning("Gemini candidate content is not an object.")
            raise AIProviderError()

        parts = content.get("parts")
        if not isinstance(parts, list) or len(parts) == 0:
            logger.warning("Gemini response has no content parts.")
            raise AIProviderError()

        text_parts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)

        if not text_parts:
            logger.warning("Gemini response has no text parts.")
            raise AIProviderError()

        text = "".join(text_parts).strip()

        if not text:
            logger.warning("Gemini returned an empty answer.")
            raise AIProviderError()

        if len(text) > MAX_ANSWER_CHARS:
            logger.warning("Gemini response exceeded the answer length limit.")
            raise AIProviderError()

        return text
