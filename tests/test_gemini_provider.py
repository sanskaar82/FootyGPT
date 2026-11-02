"""Tests for the Google Gemini AI provider, using mocked HTTP responses.

No real network requests are made and no API key is used. The placeholder key
below exists only to prove it never leaks into error messages.
"""

import asyncio
import json

import httpx
import pytest

from backend.config import Settings
from backend.exceptions import (
    AIConfigError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from backend.services.providers.gemini import GeminiProvider

TEST_KEY = "test-key-not-real"
BASE_URL = "https://test.gemini.local/v1beta"
MODEL = "gemini-1.5-flash"


def make_settings(**overrides: object) -> Settings:
    """Build settings for tests, ignoring any local ``.env`` file."""
    values: dict[str, object] = {
        "ai_provider": "gemini",
        "gemini_api_key": TEST_KEY,
        "gemini_model": MODEL,
        "gemini_base_url": BASE_URL,
        "ai_timeout_seconds": 5,
        "ai_max_tokens": 100,
        "ai_temperature": 0.3,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def ok_response(text: str = "Offside is when...") -> httpx.Response:
    """A valid Gemini generateContent response."""
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": text}],
                        "role": "model",
                    },
                    "index": 0,
                }
            ],
            "model": MODEL,
        },
    )


def make_service(handler, **overrides: object) -> GeminiProvider:
    """Create a Gemini provider whose HTTP calls go to ``handler``."""
    return GeminiProvider(make_settings(**overrides), transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_successful_response_returns_text() -> None:
    """A valid response yields the assistant's text, stripped."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("  A player is offside if... \n")

    service = make_service(handler)
    answer = await service.complete([{"role": "user", "content": "When is a player offside?"}])

    assert answer == "A player is offside if..."


@pytest.mark.anyio
async def test_request_shape() -> None:
    """The request contains the model, system instruction, history, and question."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("x-goog-api-key")
        captured["body"] = json.loads(request.content)
        return ok_response()

    service = make_service(handler)
    messages = [
        {"role": "system", "content": "You are FootyGPT"},
        {"role": "user", "content": "What is offside?"},
        {"role": "assistant", "content": "It is when..."},
        {"role": "user", "content": "Can you give an example?"},
    ]
    await service.complete(messages)

    assert captured["url"] == f"{BASE_URL}/models/{MODEL}:generateContent"
    assert captured["auth"] == f"{TEST_KEY}"
    body = captured["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "You are FootyGPT"}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "What is offside?"}]},
        {"role": "model", "parts": [{"text": "It is when..."}]},
        {"role": "user", "parts": [{"text": "Can you give an example?"}]},
    ]
    assert body["generationConfig"]["maxOutputTokens"] == 100
    assert body["generationConfig"]["temperature"] == 0.3


@pytest.mark.anyio
async def test_system_instruction_mapping() -> None:
    """System messages are combined into systemInstruction, not contents."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return ok_response()

    service = make_service(handler)
    await service.complete(
        [
            {"role": "system", "content": "System part one"},
            {"role": "system", "content": "System part two"},
            {"role": "user", "content": "Hello"},
        ]
    )

    body = captured["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "System part one\nSystem part two"}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
    ]


@pytest.mark.anyio
async def test_conversation_history_mapping() -> None:
    """User→user, assistant→model role mapping preserves ordering."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return ok_response()

    service = make_service(handler)
    await service.complete(
        [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
    )

    body = captured["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "SYSTEM"}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Question 1"}]},
        {"role": "model", "parts": [{"text": "Answer 1"}]},
        {"role": "user", "parts": [{"text": "Question 2"}]},
    ]


@pytest.mark.anyio
async def test_multiple_text_parts_are_joined() -> None:
    """Multiple content parts are concatenated into a single answer."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "A player is "}, {"text": "offside if..."}],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ]
            },
        )

    service = make_service(handler)
    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "A player is offside if..."


@pytest.mark.anyio
async def test_missing_api_key_raises_config_error() -> None:
    """No key means a clear configuration error, before any HTTP call."""
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        nonlocal called
        called = True
        return ok_response()

    service = make_service(handler, gemini_api_key=None)
    with pytest.raises(AIConfigError):
        await service.complete([{"role": "user", "content": "Hello"}])
    assert called is False


@pytest.mark.anyio
async def test_missing_model_raises_config_error() -> None:
    """No model configured is also a configuration error."""

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return ok_response()

    service = make_service(handler, gemini_model=None)
    with pytest.raises(AIConfigError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_auth_errors_raise_config_error() -> None:
    """HTTP 401 and 403 map to configuration errors."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    service = make_service(handler)
    with pytest.raises(AIConfigError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_rate_limit_raises_rate_limit_error() -> None:
    """HTTP 429 maps to a rate-limit error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    service = make_service(handler)
    with pytest.raises(AIRateLimitError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_provider_server_error_raises_provider_error() -> None:
    """A 5xx from the provider maps to a provider error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal boom")

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_network_error_raises_provider_error() -> None:
    """A connection failure maps to a provider error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_timeout_raises_timeout_error() -> None:
    """A timeout maps to a timeout error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    service = make_service(handler)
    with pytest.raises(AITimeoutError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_malformed_json_raises_provider_error() -> None:
    """Invalid JSON from Gemini maps to a provider error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_empty_response_raises_provider_error() -> None:
    """An empty candidates list maps to a provider error (e.g. safety-blocked)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_no_text_parts_raises_provider_error() -> None:
    """A candidate with no text parts maps to a provider error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ]
            },
        )

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_max_answer_chars_enforced() -> None:
    """An oversized Gemini answer is rejected, not stored."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "x" * 20001}],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ]
            },
        )

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hi"}])


@pytest.mark.anyio
async def test_api_key_never_appears_in_errors() -> None:
    """Error messages never contain the API key."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {TEST_KEY}")

    service = make_service(handler)
    with pytest.raises(AIConfigError) as excinfo:
        await service.complete([{"role": "user", "content": "Hello"}])

    assert TEST_KEY not in str(excinfo.value)


@pytest.mark.anyio
async def test_no_automatic_retry_on_failure() -> None:
    """A failing request is attempted exactly once, so costs stay predictable."""
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "hi"}])

    assert calls == 1


def test_provider_factory_selects_gemini() -> None:
    """AI_PROVIDER=gemini selects GeminiProvider via the factory."""
    from backend.services.ai_service import AIService

    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="gemini",
            gemini_api_key="key",
            gemini_model="gemini-1.5-flash",
        )
    )

    assert isinstance(service.provider, GeminiProvider)
    assert service.provider.model == "gemini-1.5-flash"
