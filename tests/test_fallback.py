"""Tests for automatic AI provider fallback.

Covers fallback configuration, retryable exception classification,
fallback triggering/suppression, LangSmith multi-attempt tracing,
and API key privacy.
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
    AIServiceError,
)
from backend.services.ai_service import AIService
from backend.services.providers.base import AIProvider
from backend.services.providers.gemini import GeminiProvider
from backend.services.providers.openrouter import OpenRouterProvider


def ok_openrouter_response(content: str = "Offside is when...") -> httpx.Response:
    """A valid OpenRouter chat-completions response."""
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "model": "test/model",
        },
    )


def ok_gemini_response(text: str = "Offside is when...") -> httpx.Response:
    """A valid Gemini generateContent response."""
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": text}], "role": "model"},
                    "index": 0,
                }
            ],
            "model": "gemini-1.5-flash",
        },
    )


# --- Configuration tests ------------------------------------------------------


def test_fallback_disabled_by_default() -> None:
    """AI_FALLBACK_PROVIDER defaults to None (fallback disabled)."""
    config = Settings(_env_file=None)

    assert config.ai_fallback_provider is None
    assert config.fallback_configured is False


def test_gemini_primary_openrouter_fallback_config() -> None:
    """Gemini primary with OpenRouter fallback is valid."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        ai_fallback_provider="openrouter",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
    )

    assert config.ai_provider == "gemini"
    assert config.ai_fallback_provider == "openrouter"
    assert config.fallback_configured is True


def test_openrouter_primary_gemini_fallback_config() -> None:
    """OpenRouter primary with Gemini fallback is valid."""
    config = Settings(
        _env_file=None,
        ai_provider="openrouter",
        ai_fallback_provider="gemini",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    assert config.ai_provider == "openrouter"
    assert config.ai_fallback_provider == "gemini"
    assert config.fallback_configured is True


def test_blank_fallback_configuration() -> None:
    """Blank AI_FALLBACK_PROVIDER is treated as disabled."""
    config = Settings(
        _env_file=None,
        ai_provider="openrouter",
        ai_fallback_provider="   ",
    )

    assert config.ai_fallback_provider is None
    assert config.fallback_configured is False


def test_unsupported_fallback_provider_rejected() -> None:
    """An unsupported fallback provider is rejected at startup."""
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            ai_provider="openrouter",
            ai_fallback_provider="anthropic",
            openrouter_api_key="key",
            openrouter_model="m",
        )


def test_same_primary_and_fallback_rejected() -> None:
    """Fallback provider must differ from primary provider."""
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            ai_provider="openrouter",
            ai_fallback_provider="openrouter",
            openrouter_api_key="key",
            openrouter_model="m",
        )


def test_fallback_not_configured_when_credentials_missing() -> None:
    """fallback_configured is False when fallback provider lacks credentials."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        ai_fallback_provider="openrouter",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
        openrouter_api_key=None,
        openrouter_model=None,
    )

    assert config.ai_fallback_provider == "openrouter"
    assert config.fallback_configured is False


# --- Exception classification tests ------------------------------------------


def test_ai_provider_error_retryable_attribute() -> None:
    """AIProviderError exposes a retryable attribute (default False)."""
    assert AIProviderError().retryable is False
    assert AIProviderError(retryable=True).retryable is True


def test_ai_config_error_not_retryable() -> None:
    """AIConfigError is never retryable."""
    assert not isinstance(AIConfigError(), AIProviderError)


def test_http_5xx_is_retryable() -> None:
    """HTTP 5xx errors create retryable AIProviderError in both providers."""

    async def check(provider_cls, config_kwargs) -> None:
        config = Settings(_env_file=None, **config_kwargs)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server error")

        provider = provider_cls(config, transport=httpx.MockTransport(handler))
        with pytest.raises(AIProviderError) as excinfo:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert excinfo.value.retryable is True

    asyncio.run(check(
        OpenRouterProvider,
        {"ai_provider": "openrouter", "openrouter_api_key": "key", "openrouter_model": "m"},
    ))
    asyncio.run(check(
        GeminiProvider,
        {"ai_provider": "gemini", "gemini_api_key": "key", "gemini_model": "gemini-1.5-flash"},
    ))


def test_http_429_is_not_ai_provider_error() -> None:
    """HTTP 429 raises AIRateLimitError, not AIProviderError."""

    async def check(provider_cls, config_kwargs) -> None:
        config = Settings(_env_file=None, **config_kwargs)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "slow down"}})

        provider = provider_cls(config, transport=httpx.MockTransport(handler))
        with pytest.raises(AIRateLimitError):
            await provider.complete([{"role": "user", "content": "Hi"}])

    asyncio.run(check(
        OpenRouterProvider,
        {"ai_provider": "openrouter", "openrouter_api_key": "key", "openrouter_model": "m"},
    ))
    asyncio.run(check(
        GeminiProvider,
        {"ai_provider": "gemini", "gemini_api_key": "key", "gemini_model": "gemini-1.5-flash"},
    ))


def test_http_401_is_config_error_not_retryable() -> None:
    """HTTP 401 raises AIConfigError, not retryable AIProviderError."""

    async def check(provider_cls, config_kwargs) -> None:
        config = Settings(_env_file=None, **config_kwargs)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "invalid key"}})

        provider = provider_cls(config, transport=httpx.MockTransport(handler))
        with pytest.raises(AIConfigError):
            await provider.complete([{"role": "user", "content": "Hi"}])

    asyncio.run(check(
        OpenRouterProvider,
        {"ai_provider": "openrouter", "openrouter_api_key": "key", "openrouter_model": "m"},
    ))
    asyncio.run(check(
        GeminiProvider,
        {"ai_provider": "gemini", "gemini_api_key": "key", "gemini_model": "gemini-1.5-flash"},
    ))


def test_timeout_is_eligible_for_fallback() -> None:
    """AITimeoutError is a distinct exception eligible for fallback."""
    error = AITimeoutError()
    assert isinstance(error, AIServiceError)


# --- Fallback behavior tests --------------------------------------------------


def _make_openrouter_service(
    handler, fallback_handler=None, **overrides
) -> AIService:
    """Build an AIService with OpenRouter primary and optional fallback."""
    values = {
        "ai_provider": "openrouter",
        "openrouter_api_key": "or-key",
        "openrouter_model": "openrouter/free",
        "langsmith_tracing": False,
        "langsmith_api_key": None,
    }
    values.update(overrides)
    config = Settings(_env_file=None, **values)

    transport = httpx.MockTransport(handler)
    fallback_transport = (
        httpx.MockTransport(fallback_handler) if fallback_handler else None
    )

    service = AIService(config, transport=transport)
    # Override fallback transport if provided
    if fallback_handler and service._fallback_provider is not None:
        service._fallback_provider.transport = fallback_transport
    return service


def _make_gemini_service(
    handler, fallback_handler=None, **overrides
) -> AIService:
    """Build an AIService with Gemini primary and optional fallback."""
    values = {
        "ai_provider": "gemini",
        "gemini_api_key": "gem-key",
        "gemini_model": "gemini-1.5-flash",
        "langsmith_tracing": False,
        "langsmith_api_key": None,
    }
    values.update(overrides)
    config = Settings(_env_file=None, **values)

    transport = httpx.MockTransport(handler)
    fallback_transport = (
        httpx.MockTransport(fallback_handler) if fallback_handler else None
    )

    service = AIService(config, transport=transport)
    if fallback_handler and service._fallback_provider is not None:
        service._fallback_provider.transport = fallback_transport
    return service


@pytest.mark.anyio
async def test_primary_succeeds_fallback_not_called() -> None:
    """When primary succeeds, fallback is not called."""
    fallback_called = False

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return ok_openrouter_response("Primary answer")

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_called
        fallback_called = True
        return ok_openrouter_response("Fallback answer")

    service = _make_openrouter_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Primary answer"
    assert fallback_called is False
    assert service.last_provider == "openrouter"
    assert service.last_model == "openrouter/free"


@pytest.mark.anyio
async def test_primary_429_triggers_fallback() -> None:
    """HTTP 429 from primary triggers fallback attempt."""
    fallback_calls = 0

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_calls
        fallback_calls += 1
        return ok_openrouter_response("Fallback answer")

    service = _make_gemini_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Fallback answer"
    assert fallback_calls == 1
    assert service.last_provider == "openrouter"
    assert service.last_model == "openrouter/free"


@pytest.mark.anyio
async def test_primary_timeout_triggers_fallback() -> None:
    """Timeout from primary triggers fallback attempt."""
    fallback_calls = 0

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_calls
        fallback_calls += 1
        return ok_gemini_response("Fallback answer")

    service = _make_openrouter_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Fallback answer"
    assert fallback_calls == 1
    assert service.last_provider == "gemini"
    assert service.last_model == "gemini-1.5-flash"


@pytest.mark.anyio
async def test_primary_5xx_triggers_fallback() -> None:
    """HTTP 5xx from primary triggers fallback attempt."""
    fallback_calls = 0

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_calls
        fallback_calls += 1
        return ok_openrouter_response("Fallback answer")

    service = _make_gemini_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Fallback answer"
    assert fallback_calls == 1
    assert service.last_provider == "openrouter"


@pytest.mark.anyio
async def test_primary_malformed_no_fallback() -> None:
    """Malformed response from primary does NOT trigger fallback."""
    fallback_called = False

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"no_choices": True})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_called
        fallback_called = True
        return ok_gemini_response("Fallback answer")

    service = _make_gemini_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
    )

    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert fallback_called is False


@pytest.mark.anyio
async def test_primary_invalid_credentials_no_fallback() -> None:
    """Invalid credentials from primary do NOT trigger fallback."""
    fallback_called = False

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_called
        fallback_called = True
        return ok_gemini_response("Fallback answer")

    service = _make_gemini_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
    )

    with pytest.raises(AIConfigError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert fallback_called is False


@pytest.mark.anyio
async def test_primary_missing_configuration_no_fallback() -> None:
    """Missing primary configuration raises AIConfigError without fallback."""
    service = _make_openrouter_service(
        lambda request: ok_openrouter_response(),
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
        openrouter_api_key=None,  # Missing primary key
    )

    with pytest.raises(AIConfigError):
        await service.complete([{"role": "user", "content": "Hi"}])


@pytest.mark.anyio
async def test_fallback_succeeds_response_returned() -> None:
    """When fallback succeeds, its response is returned."""
    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        return ok_gemini_response("Fallback response")

    service = _make_openrouter_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Fallback response"
    assert service.last_provider == "gemini"
    assert service.last_model == "gemini-1.5-flash"


@pytest.mark.anyio
async def test_both_fail_primary_exception_preserved() -> None:
    """When both providers fail, the original primary exception is raised."""
    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="fallback server error")

    service = _make_openrouter_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    with pytest.raises(AIRateLimitError) as excinfo:
        await service.complete([{"role": "user", "content": "Hi"}])

    assert isinstance(excinfo.value, AIRateLimitError)
    assert not isinstance(excinfo.value, AIProviderError)


@pytest.mark.anyio
async def test_fallback_missing_credentials_primary_preserved() -> None:
    """When fallback lacks credentials, primary exception is preserved."""
    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    service = _make_openrouter_service(
        primary_handler,
        ai_fallback_provider="gemini",
        gemini_api_key=None,
        gemini_model=None,
    )

    # Fallback provider should not be instantiated
    assert service._fallback_provider is None

    with pytest.raises(AIRateLimitError):
        await service.complete([{"role": "user", "content": "Hi"}])


@pytest.mark.anyio
async def test_fallback_attempted_at_most_once() -> None:
    """Each provider is attempted at most once per request."""
    primary_calls = 0
    fallback_calls = 0

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        nonlocal primary_calls
        primary_calls += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_calls
        fallback_calls += 1
        return httpx.Response(500, text="fallback error")

    service = _make_openrouter_service(
        primary_handler,
        fallback_handler=fallback_handler,
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    with pytest.raises(AIRateLimitError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert primary_calls == 1
    assert fallback_calls == 1


@pytest.mark.anyio
async def test_no_infinite_retry_loop() -> None:
    """No infinite retry loop: each provider attempted exactly once."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    service = _make_openrouter_service(
        handler,
        fallback_handler=handler,
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    with pytest.raises(AIRateLimitError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert call_count == 2  # Primary once, fallback once


@pytest.mark.anyio
async def test_fallback_disabled_original_behavior() -> None:
    """With fallback disabled, behavior is unchanged from before fallback."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    service = _make_openrouter_service(
        handler,
        # No ai_fallback_provider - fallback disabled
    )

    assert service._fallback_provider is None
    assert config_fallback_disabled(service)

    with pytest.raises(AIRateLimitError):
        await service.complete([{"role": "user", "content": "Hi"}])


def config_fallback_disabled(service: AIService) -> bool:
    """Helper: check that fallback is disabled on the service."""
    return service._fallback_provider is None


# --- LangSmith tracing tests --------------------------------------------------


class _FakeRun:
    def __init__(self, run_id: str = "run-123"):
        self.id = run_id


class _FakeLangSmithClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.created_runs = []
        self.updated_runs = []

    def create_run(self, **kwargs):
        self.created_runs.append(kwargs)
        return _FakeRun("run-123")

    def update_run(self, run_id=None, **kwargs):
        self.updated_runs.append((run_id, kwargs))


def _monkey_langsmith_client(monkeypatch):
    fake = _FakeLangSmithClient()

    def fake_client_factory(api_key=None):
        return fake

    monkeypatch.setattr("langsmith.Client", fake_client_factory)
    return fake


@pytest.mark.anyio
async def test_primary_success_one_trace(monkeypatch) -> None:
    """Successful primary request generates one LangSmith trace."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_openrouter_response("Primary answer")

    service = _make_openrouter_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
        ai_fallback_provider="gemini",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Primary answer"
    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1

    create_kwargs = fake.created_runs[0]
    assert create_kwargs["extra"]["metadata"]["attempt"] == 1
    assert create_kwargs["extra"]["metadata"]["attempt_type"] == "primary"

    run_id, update_kwargs = fake.updated_runs[0]
    assert update_kwargs["extra"]["metadata"]["status"] == "success"
    assert update_kwargs["extra"]["metadata"]["attempt_type"] == "primary"


@pytest.mark.anyio
async def test_primary_failure_fallback_success_two_traces(monkeypatch) -> None:
    """Primary failure and fallback success generate two traces."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        return ok_openrouter_response("Fallback answer")

    service = _make_gemini_service(
        primary_handler,
        fallback_handler=fallback_handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
        ai_fallback_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Fallback answer"
    assert len(fake.created_runs) == 2
    assert len(fake.updated_runs) == 2

    # Primary run: attempt 1, error
    primary_create = fake.created_runs[0]
    assert primary_create["extra"]["metadata"]["attempt"] == 1
    assert primary_create["extra"]["metadata"]["attempt_type"] == "primary"
    assert primary_create["extra"]["metadata"]["provider"] == "gemini"

    # Fallback run: attempt 2, success
    fallback_create = fake.created_runs[1]
    assert fallback_create["extra"]["metadata"]["attempt"] == 2
    assert fallback_create["extra"]["metadata"]["attempt_type"] == "fallback"
    assert fallback_create["extra"]["metadata"]["provider"] == "openrouter"

    primary_update = fake.updated_runs[0]
    assert primary_update[1]["extra"]["metadata"]["status"] == "error"

    fallback_update = fake.updated_runs[1]
    assert fallback_update[1]["extra"]["metadata"]["status"] == "success"


@pytest.mark.anyio
async def test_both_fail_two_failure_traces(monkeypatch) -> None:
    """Both providers failing generates two failure traces."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="fallback error")

    service = _make_gemini_service(
        primary_handler,
        fallback_handler=fallback_handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
        ai_fallback_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
    )

    with pytest.raises(AIRateLimitError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert len(fake.created_runs) == 2
    assert len(fake.updated_runs) == 2

    primary_update = fake.updated_runs[0]
    assert primary_update[1]["extra"]["metadata"]["status"] == "error"
    assert primary_update[1]["extra"]["metadata"]["attempt_type"] == "primary"

    fallback_update = fake.updated_runs[1]
    assert fallback_update[1]["extra"]["metadata"]["status"] == "error"
    assert fallback_update[1]["extra"]["metadata"]["attempt_type"] == "fallback"


@pytest.mark.anyio
async def test_langsmith_unavailable_does_not_break_ai(monkeypatch) -> None:
    """LangSmith client creation failure does not affect AI response."""
    fake = _monkey_langsmith_client(monkeypatch)

    def failing_create_run(**kwargs):
        raise RuntimeError("LangSmith is down")

    fake.create_run = failing_create_run

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_openrouter_response("Answer despite tracing failure")

    service = _make_openrouter_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Answer despite tracing failure"


@pytest.mark.anyio
async def test_api_key_never_in_traces(monkeypatch) -> None:
    """API keys never appear in LangSmith trace metadata."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_openrouter_response("Answer")

    service = _make_openrouter_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="secret-langsmith-key",
        openrouter_api_key="secret-openrouter-key",
        ai_fallback_provider="gemini",
        gemini_api_key="secret-gemini-key",
        gemini_model="gemini-1.5-flash",
    )

    await service.complete([{"role": "user", "content": "Hi"}])

    all_traces = str(fake.created_runs) + str(fake.updated_runs)
    assert "secret-langsmith-key" not in all_traces
    assert "secret-openrouter-key" not in all_traces
    assert "secret-gemini-key" not in all_traces
    assert "Authorization" not in all_traces
    assert "Bearer" not in all_traces


@pytest.mark.anyio
async def test_application_works_without_langsmith() -> None:
    """AI requests succeed normally when LangSmith is not configured."""
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return ok_openrouter_response("Answer")

    service = _make_openrouter_service(
        handler,
        langsmith_tracing=False,
        langsmith_api_key=None,
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Answer"
    assert called is True


# --- Provider type checks -----------------------------------------------------


def test_openrouter_provider_is_ai_provider() -> None:
    """OpenRouterProvider implements AIProvider interface."""
    config = Settings(
        _env_file=None,
        ai_provider="openrouter",
        openrouter_api_key="key",
        openrouter_model="m",
    )
    provider = OpenRouterProvider(config)
    assert isinstance(provider, AIProvider)


def test_gemini_provider_is_ai_provider() -> None:
    """GeminiProvider implements AIProvider interface."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        gemini_api_key="key",
        gemini_model="gemini-1.5-flash",
    )
    provider = GeminiProvider(config)
    assert isinstance(provider, AIProvider)
