"""Tests for the OpenRouter AI service, using mocked HTTP responses.

No real network requests are made and no API key is used. The placeholder key
below exists only to prove it never leaks into error messages.
"""

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
from backend.services.ai_service import AIService
from backend.services.prompts import SYSTEM_PROMPT, build_messages

TEST_KEY = "test-key-not-real"
BASE_URL = "https://test.openrouter.local/api/v1"


def make_settings(**overrides: object) -> Settings:
    """Build settings for tests, ignoring any local ``.env`` file."""
    values: dict[str, object] = {
        "openrouter_api_key": TEST_KEY,
        "openrouter_model": "test/model",
        "openrouter_base_url": BASE_URL,
        "ai_timeout_seconds": 5,
        "ai_max_tokens": 100,
        "ai_temperature": 0.3,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def ok_response(content: str = "Offside is when...") -> httpx.Response:
    """A valid OpenRouter chat-completions response."""
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "model": "test/model",
        },
    )


def make_service(handler, **overrides: object) -> AIService:
    """Create an AI service whose HTTP calls go to ``handler``."""
    return AIService(make_settings(**overrides), transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_successful_response_returns_text() -> None:
    """A valid response yields the assistant's text, stripped."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("  A player is offside if... \n")

    service = make_service(handler)
    answer = await service.complete(build_messages("When is a player offside?"))

    assert answer == "A player is offside if..."


@pytest.mark.anyio
async def test_request_shape() -> None:
    """The request contains the model, system prompt, history, and question."""
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return ok_response()

    service = make_service(handler)
    messages = build_messages(
        "Can you give an example?",
        history=[
            {"role": "user", "content": "What is offside?"},
            {"role": "assistant", "content": "It is..."},
        ],
    )
    await service.complete(messages)

    assert captured["url"] == f"{BASE_URL}/chat/completions"
    assert captured["auth"] == f"Bearer {TEST_KEY}"
    body = captured["body"]
    assert body["model"] == "test/model"
    assert body["messages"] == messages
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert body["messages"][-1]["content"] == "Can you give an example?"
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.3


@pytest.mark.anyio
async def test_missing_api_key_raises_config_error() -> None:
    """No key means a clear configuration error, before any HTTP call."""
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        nonlocal called
        called = True
        return ok_response()

    service = make_service(handler, openrouter_api_key=None)
    with pytest.raises(AIConfigError):
        await service.complete(build_messages("Hello"))
    assert called is False


@pytest.mark.anyio
async def test_missing_model_raises_config_error() -> None:
    """No model configured is also a configuration error."""

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return ok_response()

    service = make_service(handler, openrouter_model=None)
    with pytest.raises(AIConfigError):
        await service.complete(build_messages("Hello"))


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 402, 403])
async def test_auth_and_billing_errors_raise_config_error(status: int) -> None:
    """Invalid keys and billing problems map to a configuration error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    service = make_service(handler)
    with pytest.raises(AIConfigError):
        await service.complete(build_messages("Hello"))


@pytest.mark.anyio
async def test_rate_limit_raises_rate_limit_error() -> None:
    """HTTP 429 maps to a rate-limit error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    service = make_service(handler)
    with pytest.raises(AIRateLimitError):
        await service.complete(build_messages("Hello"))


@pytest.mark.anyio
async def test_provider_server_error_raises_provider_error() -> None:
    """A 5xx from the provider maps to a provider error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal boom")

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete(build_messages("Hello"))


@pytest.mark.anyio
async def test_network_error_raises_provider_error() -> None:
    """A connection failure maps to a provider error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete(build_messages("Hello"))


@pytest.mark.anyio
async def test_timeout_raises_timeout_error() -> None:
    """A timeout maps to a timeout error."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    service = make_service(handler)
    with pytest.raises(AITimeoutError):
        await service.complete(build_messages("Hello"))


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        b"not json at all",
        {"choices": []},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"message": {}}]},
        {"no_choices_key": True},
    ],
    ids=["invalid-json", "empty-choices", "empty-content", "blank-content", "no-content", "wrong-shape"],
)
async def test_malformed_response_raises_provider_error(payload: object) -> None:
    """Malformed or empty provider responses map to a provider error."""
    if isinstance(payload, bytes):

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=payload)

    else:

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

    service = make_service(handler)
    with pytest.raises(AIProviderError):
        await service.complete(build_messages("Hello"))


@pytest.mark.anyio
async def test_api_key_never_appears_in_errors() -> None:
    """Error messages never contain the API key or request headers."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {TEST_KEY}")

    service = make_service(handler)
    with pytest.raises(AIConfigError) as excinfo:
        await service.complete(build_messages("Hello"))

    assert TEST_KEY not in str(excinfo.value)
    assert "Bearer" not in str(excinfo.value)


def test_build_messages_without_history() -> None:
    """A new conversation sends just the system prompt and question."""
    messages = build_messages("What is a high press?")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[-1]["content"] == "What is a high press?"
