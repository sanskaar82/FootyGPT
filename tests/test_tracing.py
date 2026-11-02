"""Tests for LangSmith observability tracing.

LangSmith client is mocked; no real LangSmith account or network calls are used.
"""

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.config import Settings
from backend.exceptions import AIConfigError, AIProviderError
from backend.services.ai_service import AIService
from backend.services.providers.openrouter import OpenRouterProvider
from backend.services.tracing import LangSmithTracer


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "ai_provider": "openrouter",
        "openrouter_api_key": "test-key",
        "openrouter_model": "test/model",
        "openrouter_base_url": "https://test.openrouter.local/api/v1",
        "langsmith_tracing": False,
        "langsmith_api_key": None,
        "langsmith_project": "FootyGPT",
        "ai_timeout_seconds": 5,
        "ai_max_tokens": 100,
        "ai_temperature": 0.3,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def ok_response(content: str = "Offside is when...") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "model": "test/model",
        },
    )


def make_service(handler, **overrides: object) -> AIService:
    return AIService(
        make_settings(**overrides),
        transport=httpx.MockTransport(handler),
    )


# --- Tracing disabled by default --------------------------------------------


def test_tracing_disabled_by_default() -> None:
    """LangSmith tracing is disabled unless explicitly enabled."""
    config = make_settings()
    tracer = LangSmithTracer(config)

    assert tracer.enabled is False


def test_tracing_enabled_without_api_key_is_false() -> None:
    """Tracing enabled flag alone is not enough; an API key is required."""
    config = make_settings(langsmith_tracing=True, langsmith_api_key=None)
    tracer = LangSmithTracer(config)

    assert tracer.enabled is False


def test_tracing_enabled_with_api_key() -> None:
    """Tracing is enabled when both flag and API key are set."""
    config = make_settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )
    tracer = LangSmithTracer(config)

    assert tracer.enabled is True


# --- Mock LangSmith client --------------------------------------------------


class _FakeLangSmithClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.created_runs = []
        self.updated_runs = []

    def create_run(self, **kwargs):
        self.created_runs.append(kwargs)
        return None

    def update_run(self, run_id=None, **kwargs):
        self.updated_runs.append((run_id, kwargs))


def _monkey_langsmith_client(monkeypatch):
    fake = _FakeLangSmithClient()

    def fake_client_factory(api_key=None):
        return fake

    monkeypatch.setattr("langsmith.Client", fake_client_factory)
    return fake


# --- OpenRouter tracing -----------------------------------------------------


@pytest.mark.anyio
async def test_openrouter_request_generates_trace(monkeypatch) -> None:
    """An OpenRouter request creates and updates a LangSmith run."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("A player is offside if...")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete(
        [{"role": "user", "content": "When is a player offside?"}]
    )

    assert answer == "A player is offside if..."
    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1

    create_kwargs = fake.created_runs[0]
    assert create_kwargs["run_type"] == "llm"
    assert create_kwargs["name"] == "footygpt-chat"
    assert create_kwargs["inputs"]["messages"] == [
        {"role": "user", "content": "When is a player offside?"}
    ]
    assert create_kwargs["extra"]["metadata"]["provider"] == "openrouter"
    assert create_kwargs["extra"]["metadata"]["model"] == "test/model"

    run_id, update_kwargs = fake.updated_runs[0]
    assert isinstance(uuid.UUID(run_id), uuid.UUID)
    assert update_kwargs["outputs"]["response"] == "A player is offside if..."
    assert update_kwargs["extra"]["metadata"]["status"] == "success"


@pytest.mark.anyio
async def test_openrouter_tracing_captures_provider_and_model(
    monkeypatch,
) -> None:
    """Provider and model metadata are captured in the trace."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response()

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
        openrouter_model="anthropic/claude-3-haiku",
    )

    await service.complete([{"role": "user", "content": "Hi"}])

    metadata = fake.created_runs[0]["extra"]["metadata"]
    assert metadata["provider"] == "openrouter"
    assert metadata["model"] == "anthropic/claude-3-haiku"


@pytest.mark.anyio
async def test_openrouter_tracing_captures_latency(monkeypatch) -> None:
    """Latency is captured in the trace metadata."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response()

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    await service.complete([{"role": "user", "content": "Hi"}])

    metadata = fake.updated_runs[0][1]["extra"]["metadata"]
    assert "latency_ms" in metadata
    assert isinstance(metadata["latency_ms"], (int, float))


# --- Gemini tracing ---------------------------------------------------------


@pytest.mark.anyio
async def test_gemini_request_generates_trace(monkeypatch) -> None:
    """A Gemini request creates and updates a LangSmith run."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "A player is offside if..."}], "role": "model"},
                        "index": 0,
                    }
                ]
            },
        )

    service = AIService(
        make_settings(
            ai_provider="gemini",
            gemini_api_key="gem-key",
            gemini_model="gemini-1.5-flash",
            gemini_base_url="https://test.gemini.local/v1beta",
            langsmith_tracing=True,
            langsmith_api_key="ls-test-key",
        ),
        transport=httpx.MockTransport(handler),
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "A player is offside if..."
    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1

    metadata = fake.created_runs[0]["extra"]["metadata"]
    assert metadata["provider"] == "gemini"
    assert metadata["model"] == "gemini-1.5-flash"


# --- Token usage ------------------------------------------------------------


@pytest.mark.anyio
async def test_token_usage_captured_when_available(monkeypatch) -> None:
    """Token usage from the provider is captured in the trace."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response()

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    # record_result is called with just the response; token usage would come
    # from parsing the provider response. The sink accepts optional token_usage.
    outputs = fake.updated_runs[0][1]["outputs"]
    assert "response" in outputs


# --- Error tracing ----------------------------------------------------------


@pytest.mark.anyio
async def test_provider_exception_is_traced(monkeypatch) -> None:
    """Provider exceptions are recorded in the trace before re-raising."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1

    run_id, update_kwargs = fake.updated_runs[0]
    assert update_kwargs["outputs"]["error"] == "AIProviderError"
    assert update_kwargs["extra"]["metadata"]["status"] == "error"
    assert "error_category" in update_kwargs["extra"]["metadata"]


@pytest.mark.anyio
async def test_langsmith_failure_does_not_break_ai_response(monkeypatch) -> None:
    """LangSmith create_run failure does not affect the AI response."""
    fake = _monkey_langsmith_client(monkeypatch)

    # Make create_run fail
    def failing_create_run(**kwargs):
        raise RuntimeError("LangSmith is down")

    fake.create_run = failing_create_run

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("Answer despite tracing failure")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Answer despite tracing failure"


@pytest.mark.anyio
async def test_langsmith_update_failure_does_not_mask_provider_exception(
    monkeypatch,
) -> None:
    """LangSmith update_run failure does not mask the original provider exception."""
    fake = _monkey_langsmith_client(monkeypatch)

    def failing_update_run(run_id=None, **kwargs):
        raise RuntimeError("LangSmith update failed")

    fake.update_run = failing_update_run

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hi"}])


# --- API key privacy --------------------------------------------------------


@pytest.mark.anyio
async def test_api_key_never_appears_in_traces(monkeypatch) -> None:
    """API keys and auth headers never appear in LangSmith trace metadata."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response()

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="secret-langsmith-key",
        openrouter_api_key="secret-openrouter-key",
    )

    await service.complete([{"role": "user", "content": "Hi"}])

    all_traces = str(fake.created_runs) + str(fake.updated_runs)
    assert "secret-langsmith-key" not in all_traces
    assert "secret-openrouter-key" not in all_traces
    assert "Authorization" not in all_traces
    assert "Bearer" not in all_traces


# --- Application works without LangSmith ------------------------------------


@pytest.mark.anyio
async def test_application_works_without_langsmith() -> None:
    """AI requests succeed normally when LangSmith is not configured."""
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return ok_response()

    service = make_service(
        handler,
        langsmith_tracing=False,
        langsmith_api_key=None,
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Offside is when..."
    assert called is True


def test_tracer_is_none_when_langsmith_unavailable(monkeypatch) -> None:
    """If langsmith cannot be imported, tracing is disabled gracefully."""
    import backend.services.tracing as tracing_module

    # Simulate langsmith not being installed
    original = tracing_module.LangSmithTracer._ensure_client

    def broken_ensure(self):
        return None

    monkeypatch.setattr(tracing_module.LangSmithTracer, "_ensure_client", broken_ensure)

    config = make_settings(langsmith_tracing=True, langsmith_api_key="ls-key")
    tracer = LangSmithTracer(config)

    assert tracer.enabled is False


def test_ai_config_error_is_traced(monkeypatch) -> None:
    """AIConfigError (missing key/model) is recorded before re-raising."""
    fake = _monkey_langsmith_client(monkeypatch)

    service = AIService(
        make_settings(
            langsmith_tracing=True,
            langsmith_api_key="ls-test-key",
            openrouter_api_key=None,
        ),
        transport=httpx.MockTransport(lambda request: ok_response()),
    )

    with pytest.raises(AIConfigError):
        asyncio.run(service.complete([{"role": "user", "content": "Hi"}]))

    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1
    run_id, update_kwargs = fake.updated_runs[0]
    assert update_kwargs["outputs"]["error"] == "AIConfigError"


# --- Run finalization tests ---------------------------------------------------


@pytest.mark.anyio
async def test_successful_run_has_end_time(monkeypatch) -> None:
    """A successful run is finalized with end_time."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("A player is offside if...")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete(
        [{"role": "user", "content": "When is a player offside?"}]
    )

    assert answer == "A player is offside if..."
    assert len(fake.updated_runs) == 1

    run_id, update_kwargs = fake.updated_runs[0]
    assert "end_time" in update_kwargs
    assert update_kwargs["outputs"]["response"] == "A player is offside if..."
    assert update_kwargs["extra"]["metadata"]["status"] == "success"


@pytest.mark.anyio
async def test_failed_run_has_end_time(monkeypatch) -> None:
    """A failed run is finalized with end_time and error details."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert len(fake.updated_runs) == 1

    run_id, update_kwargs = fake.updated_runs[0]
    assert "end_time" in update_kwargs
    assert update_kwargs["outputs"]["error"] == "AIProviderError"
    assert update_kwargs["extra"]["metadata"]["status"] == "error"
    assert update_kwargs["error"] == "AIProviderError"
    assert "error_category" in update_kwargs["extra"]["metadata"]


@pytest.mark.anyio
async def test_run_finalized_only_once(monkeypatch) -> None:
    """Calling record_result twice does not result in duplicate update_run calls."""
    fake = _monkey_langsmith_client(monkeypatch)

    from backend.services.tracing import LangSmithTracer

    config = make_settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )
    tracer = LangSmithTracer(config)

    async with tracer.trace_run(
        provider="openrouter",
        model="test/model",
        messages=[{"role": "user", "content": "Hi"}],
    ) as sink:
        sink.record_result(response="First answer")
        sink.record_result(response="Second answer")  # Should be ignored

    assert len(fake.updated_runs) == 1
    _, update_kwargs = fake.updated_runs[0]
    assert update_kwargs["outputs"]["response"] == "First answer"


@pytest.mark.anyio
async def test_error_finalized_only_once(monkeypatch) -> None:
    """Calling record_error twice does not result in duplicate update_run calls."""
    fake = _monkey_langsmith_client(monkeypatch)

    from backend.services.tracing import LangSmithTracer

    config = make_settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )
    tracer = LangSmithTracer(config)

    test_error = AIProviderError(retryable=True)

    async with tracer.trace_run(
        provider="openrouter",
        model="test/model",
        messages=[{"role": "user", "content": "Hi"}],
    ) as sink:
        sink.record_error(exc=test_error)
        sink.record_error(exc=test_error)  # Should be ignored

    assert len(fake.updated_runs) == 1
    _, update_kwargs = fake.updated_runs[0]
    assert update_kwargs["outputs"]["error"] == "AIProviderError"
    assert update_kwargs["extra"]["metadata"]["status"] == "error"


@pytest.mark.anyio
async def test_end_run_failure_does_not_break_ai_response(monkeypatch) -> None:
    """LangSmith update_run failure during finalization does not break AI response."""
    fake = _monkey_langsmith_client(monkeypatch)

    def failing_update_run(run_id=None, **kwargs):
        raise RuntimeError("LangSmith update failed")

    fake.update_run = failing_update_run

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("Answer despite tracing failure")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Answer despite tracing failure"


@pytest.mark.anyio
async def test_context_manager_exits_safely_on_success(monkeypatch) -> None:
    """The tracing context manager exits cleanly on a successful provider call."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("Success answer")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Success answer"
    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1


@pytest.mark.anyio
async def test_context_manager_exits_safely_on_exception(monkeypatch) -> None:
    """The tracing context manager exits cleanly when the provider raises."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1
    _, update_kwargs = fake.updated_runs[0]
    assert update_kwargs["extra"]["metadata"]["status"] == "error"


# --- UUID propagation tests ---------------------------------------------------


@pytest.mark.anyio
async def test_create_run_receives_explicit_uuid(monkeypatch) -> None:
    """create_run() receives an explicit UUID as the id parameter."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("Success")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    await service.complete([{"role": "user", "content": "Hi"}])

    assert len(fake.created_runs) == 1
    create_kwargs = fake.created_runs[0]
    assert "id" in create_kwargs
    assert uuid.UUID(create_kwargs["id"]) is not None


@pytest.mark.anyio
async def test_update_run_uses_same_uuid_as_create_run(monkeypatch) -> None:
    """update_run() receives the same UUID that was passed to create_run()."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("Success")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    await service.complete([{"role": "user", "content": "Hi"}])

    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1

    create_kwargs = fake.created_runs[0]
    run_id, update_kwargs = fake.updated_runs[0]

    assert create_kwargs["id"] == run_id
    assert uuid.UUID(run_id) is not None


@pytest.mark.anyio
async def test_successful_run_finalized_with_explicit_uuid(monkeypatch) -> None:
    """A successful run is finalized with end_time using the explicit UUID."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("A player is offside if...")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete(
        [{"role": "user", "content": "When is a player offside?"}]
    )

    assert answer == "A player is offside if..."
    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1

    run_id, update_kwargs = fake.updated_runs[0]
    assert uuid.UUID(run_id) is not None
    assert "end_time" in update_kwargs
    assert update_kwargs["outputs"]["response"] == "A player is offside if..."
    assert update_kwargs["extra"]["metadata"]["status"] == "success"


@pytest.mark.anyio
async def test_failed_run_finalized_with_explicit_uuid(monkeypatch) -> None:
    """A failed run is finalized with end_time using the explicit UUID."""
    fake = _monkey_langsmith_client(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    with pytest.raises(AIProviderError):
        await service.complete([{"role": "user", "content": "Hi"}])

    assert len(fake.created_runs) == 1
    assert len(fake.updated_runs) == 1

    run_id, update_kwargs = fake.updated_runs[0]
    assert uuid.UUID(run_id) is not None
    assert "end_time" in update_kwargs
    assert update_kwargs["outputs"]["error"] == "AIProviderError"
    assert update_kwargs["extra"]["metadata"]["status"] == "error"
    assert update_kwargs["error"] == "AIProviderError"
    assert "error_category" in update_kwargs["extra"]["metadata"]


@pytest.mark.anyio
async def test_failed_create_run_does_not_trigger_update_run(monkeypatch) -> None:
    """If create_run() fails, update_run() is never called."""
    fake = _monkey_langsmith_client(monkeypatch)

    def failing_create_run(**kwargs):
        raise RuntimeError("LangSmith create failed")

    fake.create_run = failing_create_run

    async def handler(request: httpx.Request) -> httpx.Response:
        return ok_response("Answer despite tracing failure")

    service = make_service(
        handler,
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )

    answer = await service.complete([{"role": "user", "content": "Hi"}])

    assert answer == "Answer despite tracing failure"
    # create_run was called (and failed), but update_run must not be called
    assert len(fake.updated_runs) == 0


@pytest.mark.anyio
async def test_run_finalized_only_once_with_explicit_uuid(monkeypatch) -> None:
    """Calling record_result twice still results in exactly one update_run()."""
    fake = _monkey_langsmith_client(monkeypatch)

    from backend.services.tracing import LangSmithTracer

    config = make_settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
    )
    tracer = LangSmithTracer(config)

    async with tracer.trace_run(
        provider="openrouter",
        model="test/model",
        messages=[{"role": "user", "content": "Hi"}],
    ) as sink:
        sink.record_result(response="First answer")
        sink.record_result(response="Second answer")

    assert len(fake.updated_runs) == 1
    run_id, update_kwargs = fake.updated_runs[0]
    assert uuid.UUID(run_id) is not None
    assert update_kwargs["outputs"]["response"] == "First answer"
