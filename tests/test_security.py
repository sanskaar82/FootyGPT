"""Security-focused tests: error bodies, secret handling, and safe failures."""

import asyncio
from collections.abc import Generator

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from backend.config import Settings
from backend.database import get_db
from backend.exceptions import (
    AIConfigError,
    AIProviderError,
    AIServiceError,
    AIRateLimitError,
    AITimeoutError,
    DATABASE_ERROR_MESSAGE,
    GENERIC_ERROR_MESSAGE,
)
from backend.main import app
from backend.models import Message
from backend.services.ai_service import MAX_ANSWER_CHARS, AIService, get_ai_service
from backend.services.providers.gemini import GeminiProvider
from backend.services.providers.openrouter import OpenRouterProvider

SECRET = "sk-or-v1-this-is-a-fake-secret-used-only-in-tests"


@pytest.fixture()
def unsafe_client() -> Generator[TestClient, None, None]:
    """Client that returns 500 responses instead of re-raising exceptions."""
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def make_service(handler, **overrides: object) -> AIService:
    """Build an AI service whose HTTP calls go to ``handler``."""
    values: dict[str, object] = {
        "openrouter_api_key": "test-key",
        "openrouter_model": "test/model",
        "openrouter_base_url": "https://example.test/api",
    }
    values.update(overrides)
    return AIService(
        Settings(_env_file=None, **values), transport=httpx.MockTransport(handler)
    )


# --- Unexpected errors ------------------------------------------------------


def test_unexpected_exception_returns_generic_500(unsafe_client: TestClient) -> None:
    """An unexpected error returns a generic message, not internals."""

    def explode() -> Generator[Session, None, None]:
        raise RuntimeError(f"internal secret {SECRET}")
        yield  # pragma: no cover

    app.dependency_overrides[get_db] = explode
    response = unsafe_client.get("/conversations")

    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR_MESSAGE}
    assert SECRET not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text


def test_database_error_returns_friendly_503(unsafe_client: TestClient) -> None:
    """A SQLAlchemy error returns 503 without exposing SQL or paths."""

    def failing_db() -> Generator[Session, None, None]:
        raise OperationalError("SELECT 1", {}, Exception(f"no such table {SECRET}"))
        yield  # pragma: no cover

    app.dependency_overrides[get_db] = failing_db
    response = unsafe_client.get("/conversations")

    assert response.status_code == 503
    assert response.json() == {"detail": DATABASE_ERROR_MESSAGE}
    assert SECRET not in response.text
    assert "no such table" not in response.text


def test_failed_commit_saves_nothing(
    unsafe_client: TestClient, fake_ai, session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database failure while saving an exchange leaves no partial rows."""
    from backend.database import get_db as real_get_db

    def override() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[real_get_db] = override
    app.dependency_overrides[get_ai_service] = lambda: fake_ai

    def broken_commit(self: Session) -> None:
        raise OperationalError("INSERT", {}, Exception(f"disk full {SECRET}"))

    monkeypatch.setattr(Session, "commit", broken_commit, raising=True)
    response = unsafe_client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert SECRET not in response.text
    assert "INSERT" not in response.text

    monkeypatch.undo()
    with session_factory() as check:
        assert check.query(Message).count() == 0


# --- Secrets ----------------------------------------------------------------


def test_settings_never_expose_key_in_repr() -> None:
    """The settings object does not print the API key."""
    config = Settings(_env_file=None, openrouter_api_key=SECRET, openrouter_model="m")

    assert SECRET not in repr(config)
    assert SECRET not in str(config)
    assert config.openrouter_api_key_value == SECRET
    assert config.ai_configured is True


def test_missing_key_error_contains_no_secret(caplog: pytest.LogCaptureFixture) -> None:
    """A configuration error's text and logs contain no key material."""
    from backend.exceptions import AIConfigError

    service = make_service(lambda request: httpx.Response(200), openrouter_api_key=None)
    with caplog.at_level("DEBUG"):
        with pytest.raises(AIConfigError) as excinfo:
            asyncio.run(service.complete([{"role": "user", "content": "hi"}]))

    assert SECRET not in str(excinfo.value)
    assert SECRET not in caplog.text


def test_rejected_key_does_not_leak_into_errors() -> None:
    """A 401 whose body echoes the key still produces a clean error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid key {SECRET}")

    service = make_service(handler, openrouter_api_key=SECRET)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(service.complete([{"role": "user", "content": "hi"}]))

    assert SECRET not in str(excinfo.value)
    assert "Bearer" not in str(excinfo.value)


def test_error_responses_have_no_auth_headers(unsafe_client: TestClient, fake_ai) -> None:
    """Provider errors never include request headers in the response."""
    fake_ai.answer = AIProviderError()
    app.dependency_overrides[get_ai_service] = lambda: fake_ai

    response = unsafe_client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 502
    serialized = str(dict(response.headers)).lower()
    assert "authorization" not in serialized
    assert "bearer" not in serialized


# --- Input validation -------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": None},
        {"message": 123},
        {"message": ["a"]},
        {"message": "x" * 2001},
        {"message": "hi", "conversation_id": "abc"},
        {"message": "hi", "conversation_id": 0},
        {"message": "hi", "conversation_id": -5},
        {"message": "hi", "conversation_id": 1.5},
    ],
    ids=[
        "missing",
        "empty",
        "blank",
        "null",
        "int",
        "list",
        "too-long",
        "non-int-id",
        "zero-id",
        "negative-id",
        "float-id",
    ],
)
def test_chat_rejects_invalid_payloads(client: TestClient, payload: object) -> None:
    """Invalid payloads are rejected with 422 and never reach the AI."""
    assert client.post("/chat", json=payload).status_code == 422


def test_chat_rejects_malformed_json_body(client: TestClient) -> None:
    """A body that is not valid JSON returns 422."""
    response = client.post(
        "/chat", content=b"not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


@pytest.mark.parametrize("conversation_id", [1, 2, 999, 10**9])
def test_missing_conversations_are_404(client: TestClient, conversation_id: int) -> None:
    """Unknown conversation ids return 404 everywhere."""
    assert client.get(f"/conversations/{conversation_id}").status_code == 404
    assert (
        client.post(
            "/chat", json={"message": "hi", "conversation_id": conversation_id}
        ).status_code
        == 404
    )


def test_conversations_return_only_their_own_messages(client: TestClient) -> None:
    """Each conversation returns only its own messages."""
    first = client.post("/conversations", json={"title": "A"}).json()["id"]
    second = client.post("/conversations", json={"title": "B"}).json()["id"]
    client.post("/chat", json={"message": "Only in A", "conversation_id": first})
    client.post("/chat", json={"message": "Only in B", "conversation_id": second})

    contents_a = {m["content"] for m in client.get(f"/conversations/{first}").json()["messages"]}
    contents_b = {m["content"] for m in client.get(f"/conversations/{second}").json()["messages"]}

    assert "Only in B" not in contents_a
    assert "Only in A" not in contents_b


# --- AI response safety -----------------------------------------------------


def test_oversized_ai_answer_is_rejected() -> None:
    """A huge provider reply is treated as an error, not stored."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "x" * (MAX_ANSWER_CHARS + 1)}}]}
        )

    with pytest.raises(AIProviderError):
        asyncio.run(make_service(handler).complete([{"role": "user", "content": "hi"}]))


def test_answer_at_limit_is_accepted() -> None:
    """A reply exactly at the size limit is still allowed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "x" * MAX_ANSWER_CHARS}}]}
        )

    answer = asyncio.run(make_service(handler).complete([{"role": "user", "content": "hi"}]))
    assert len(answer) == MAX_ANSWER_CHARS


def test_no_automatic_retry_on_failure() -> None:
    """A failing request is attempted exactly once, so costs stay predictable."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with pytest.raises(AIProviderError):
        asyncio.run(make_service(handler).complete([{"role": "user", "content": "hi"}]))

    assert calls == 1


# --- Configuration safety ---------------------------------------------------


def test_blank_key_is_treated_as_missing() -> None:
    """An empty OPENROUTER_API_KEY does not count as configured."""
    config = Settings(_env_file=None, openrouter_api_key="   ", openrouter_model="m")

    assert config.openrouter_api_key is None
    assert config.ai_configured is False


def test_non_http_base_url_is_rejected() -> None:
    """A base URL that would send the key elsewhere is refused at startup."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, openrouter_base_url="ftp://evil.example")


def test_base_url_trailing_slash_is_normalised() -> None:
    """A trailing slash would create a double slash in request paths."""
    config = Settings(_env_file=None, openrouter_base_url="https://example.test/api/")

    assert config.openrouter_base_url == "https://example.test/api"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ai_timeout_seconds", 0),
        ("ai_timeout_seconds", 1000),
        ("ai_max_tokens", 0),
        ("ai_temperature", 5),
        ("ai_max_history_messages", -1),
    ],
)
def test_out_of_range_ai_settings_are_rejected(field: str, value: object) -> None:
    """Silly values in .env fail at startup instead of causing runaway costs."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


# --- Provider configuration ---------------------------------------------------


def test_ai_provider_defaults_to_openrouter() -> None:
    """The default AI provider is OpenRouter for backward compatibility."""
    config = Settings(_env_file=None)

    assert config.ai_provider == "openrouter"


def test_ai_provider_accepts_gemini() -> None:
    """Gemini is a supported AI provider."""
    config = Settings(_env_file=None, ai_provider="gemini")

    assert config.ai_provider == "gemini"


def test_ai_provider_rejects_unknown_provider() -> None:
    """An unsupported AI provider is rejected at startup."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_provider="anthropic")


def test_blank_gemini_key_is_treated_as_missing() -> None:
    """An empty GEMINI_API_KEY does not count as configured."""
    config = Settings(
        _env_file=None, ai_provider="gemini", gemini_api_key="   ", gemini_model="m"
    )

    assert config.gemini_api_key is None
    assert config.ai_configured is False


def test_gemini_configured_when_key_and_model_set() -> None:
    """ai_configured is True for Gemini when both key and model are set."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        gemini_api_key="key",
        gemini_model="gemini-1.5-flash",
    )

    assert config.ai_configured is True


def test_openrouter_configured_still_works_by_default() -> None:
    """ai_configured remains True for OpenRouter when both key and model are set."""
    config = Settings(_env_file=None, openrouter_api_key="key", openrouter_model="m")

    assert config.ai_provider == "openrouter"
    assert config.ai_configured is True


def test_gemini_not_configured_when_key_missing() -> None:
    """ai_configured is False for Gemini when the key is missing."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        gemini_model="m",
        gemini_api_key=None,
        langsmith_tracing=False,
        langsmith_api_key=None,
    )

    assert config.ai_configured is False


def test_gemini_not_configured_when_model_missing() -> None:
    """ai_configured is False for Gemini when the model is missing."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        gemini_api_key="key",
        gemini_model=None,
        langsmith_tracing=False,
        langsmith_api_key=None,
    )

    assert config.ai_configured is False


def test_provider_configured_reports_per_provider() -> None:
    """provider_configured reports readiness for each provider independently."""
    config = Settings(
        _env_file=None,
        openrouter_api_key="or-key",
        openrouter_model="or-model",
        gemini_api_key="gem-key",
        gemini_model="gem-model",
    )

    assert config.provider_configured("openrouter") is True
    assert config.provider_configured("gemini") is True


def test_gemini_api_key_never_appears_in_repr() -> None:
    """The settings object does not print the Gemini API key."""
    config = Settings(_env_file=None, ai_provider="gemini", gemini_api_key="secret-key")

    assert "secret-key" not in repr(config)
    assert "secret-key" not in str(config)
    assert config.gemini_api_key_value == "secret-key"


def test_gemini_base_url_must_be_http() -> None:
    """A Gemini base URL that would send the key elsewhere is refused at startup."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_provider="gemini", gemini_base_url="ftp://evil.example")


def test_gemini_base_url_trailing_slash_is_normalised() -> None:
    """A trailing slash on the Gemini base URL is removed."""
    config = Settings(
        _env_file=None, ai_provider="gemini", gemini_base_url="https://example.test/api/"
    )

    assert config.gemini_base_url == "https://example.test/api"


# --- Provider selection -------------------------------------------------------


def test_ai_service_defaults_to_openrouter_provider() -> None:
    """Without AI_PROVIDER, AIService uses OpenRouterProvider."""
    service = AIService(
        Settings(_env_file=None, openrouter_api_key="key", openrouter_model="m")
    )

    assert isinstance(service.provider, OpenRouterProvider)
    assert service.provider.model == "m"


def test_ai_service_explicit_openrouter_provider() -> None:
    """AI_PROVIDER=openrouter selects OpenRouterProvider."""
    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="openrouter",
            openrouter_api_key="key",
            openrouter_model="m",
        )
    )

    assert isinstance(service.provider, OpenRouterProvider)


def test_ai_service_gemini_selects_gemini_provider() -> None:
    """AI_PROVIDER=gemini selects GeminiProvider."""
    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="gemini",
            gemini_api_key="key",
            gemini_model="gemini-1.5-flash",
        )
    )

    from backend.services.providers.gemini import GeminiProvider

    assert isinstance(service.provider, GeminiProvider)
    assert service.provider.model == "gemini-1.5-flash"


def test_ai_service_missing_gemini_key_raises_config_error() -> None:
    """Missing Gemini API key raises AIConfigError before any HTTP call."""
    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="gemini",
            gemini_model="gemini-1.5-flash",
            gemini_api_key=None,
            langsmith_tracing=False,
            langsmith_api_key=None,
        )
    )

    with pytest.raises(AIConfigError):
        asyncio.run(service.complete([{"role": "user", "content": "hi"}]))


def test_ai_service_missing_gemini_model_raises_config_error() -> None:
    """Missing Gemini model raises AIConfigError before any HTTP call."""
    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="gemini",
            gemini_api_key="key",
            gemini_model=None,
            langsmith_tracing=False,
            langsmith_api_key=None,
        )
    )

    with pytest.raises(AIConfigError):
        asyncio.run(service.complete([{"role": "user", "content": "hi"}]))


def test_ai_service_missing_openrouter_key_raises_config_error() -> None:
    """Missing OpenRouter API key raises AIConfigError before any HTTP call."""
    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="openrouter",
            openrouter_model="m",
            openrouter_api_key=None,
            langsmith_tracing=False,
            langsmith_api_key=None,
        )
    )

    with pytest.raises(AIConfigError):
        asyncio.run(service.complete([{"role": "user", "content": "hi"}]))


def test_ai_service_missing_openrouter_model_raises_config_error() -> None:
    """Missing OpenRouter model raises AIConfigError before any HTTP call."""
    service = AIService(
        Settings(_env_file=None, ai_provider="openrouter", openrouter_api_key="key")
    )

    with pytest.raises(AIConfigError):
        asyncio.run(service.complete([{"role": "user", "content": "hi"}]))


def test_ai_service_unsupported_provider_raises_config_error() -> None:
    """An unsupported AI provider is rejected at Settings validation."""
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            ai_provider="anthropic",
            openrouter_api_key="key",
            openrouter_model="m",
        )


def test_ai_service_provider_selection_does_not_expose_key() -> None:
    """Provider selection constructs GeminiProvider without exposing the key."""
    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="gemini",
            gemini_api_key="secret-key",
            gemini_model="gemini-1.5-flash",
        )
    )

    assert isinstance(service.provider, GeminiProvider)
    assert "secret-key" not in repr(service.provider)


# --- Free-tier model configuration --------------------------------------------


def test_openrouter_free_model_identifier_is_accepted() -> None:
    """OpenRouter supports the openrouter/free model identifier."""
    config = Settings(
        _env_file=None,
        ai_provider="openrouter",
        openrouter_api_key="key",
        openrouter_model="openrouter/free",
    )

    assert config.openrouter_model == "openrouter/free"
    assert config.ai_configured is True


def test_openrouter_free_model_with_suffix_is_accepted() -> None:
    """OpenRouter supports model identifiers with :free suffix."""
    config = Settings(
        _env_file=None,
        ai_provider="openrouter",
        openrouter_api_key="key",
        openrouter_model="meta-llama/llama-3.1-8b-instruct:free",
    )

    assert config.openrouter_model == "meta-llama/llama-3.1-8b-instruct:free"
    assert config.ai_configured is True


def test_gemini_free_model_identifier_is_accepted() -> None:
    """Gemini supports configurable free-tier model identifiers."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        gemini_api_key="key",
        gemini_model="gemini-1.5-flash",
    )

    assert config.gemini_model == "gemini-1.5-flash"
    assert config.ai_configured is True


def test_provider_switching_gemini_to_openrouter() -> None:
    """Switching from Gemini to OpenRouter uses OpenRouter settings."""
    config = Settings(
        _env_file=None,
        ai_provider="openrouter",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    assert config.ai_provider == "openrouter"
    assert config.ai_configured is True
    assert config.provider_configured("openrouter") is True
    assert config.provider_configured("gemini") is True


def test_provider_switching_openrouter_to_gemini() -> None:
    """Switching from OpenRouter to Gemini uses Gemini settings."""
    config = Settings(
        _env_file=None,
        ai_provider="gemini",
        openrouter_api_key="or-key",
        openrouter_model="openrouter/free",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-flash",
    )

    assert config.ai_provider == "gemini"
    assert config.ai_configured is True
    assert config.provider_configured("gemini") is True


def test_missing_openrouter_model_raises_config_error_in_service() -> None:
    """Missing OpenRouter model raises AIConfigError before any HTTP call."""
    service = AIService(
        Settings(_env_file=None, ai_provider="openrouter", openrouter_api_key="key")
    )

    with pytest.raises(AIConfigError):
        asyncio.run(service.complete([{"role": "user", "content": "hi"}]))


def test_missing_gemini_model_raises_config_error_in_service() -> None:
    """Missing Gemini model raises AIConfigError before any HTTP call."""
    service = AIService(
        Settings(
            _env_file=None,
            ai_provider="gemini",
            gemini_api_key="key",
            gemini_model=None,
            langsmith_tracing=False,
            langsmith_api_key=None,
        )
    )

    with pytest.raises(AIConfigError):
        asyncio.run(service.complete([{"role": "user", "content": "hi"}]))


def test_token_configuration_is_shared_across_providers() -> None:
    """AI_MAX_TOKENS, AI_TEMPERATURE, AI_MAX_HISTORY_MESSAGES work for both providers."""
    config = Settings(
        _env_file=None,
        ai_timeout_seconds=15.0,
        ai_max_history_messages=5,
        ai_max_tokens=400,
        ai_temperature=0.7,
    )

    assert config.ai_timeout_seconds == 15.0
    assert config.ai_max_history_messages == 5
    assert config.ai_max_tokens == 400
    assert config.ai_temperature == 0.7


def test_out_of_range_token_settings_are_rejected() -> None:
    """Invalid token configuration values fail at startup."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_max_tokens=0)

    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_max_tokens=5000)

    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_temperature=5)

    with pytest.raises(ValueError):
        Settings(_env_file=None, ai_max_history_messages=-1)


# --- Retryable exception classification ---------------------------------------


def test_ai_provider_error_default_not_retryable() -> None:
    """AIProviderError defaults to retryable=False."""
    error = AIProviderError()
    assert error.retryable is False


def test_ai_provider_error_retryable_true() -> None:
    """AIProviderError can be constructed with retryable=True."""
    error = AIProviderError(retryable=True)
    assert error.retryable is True


def test_ai_provider_error_retryable_false_explicit() -> None:
    """AIProviderError(retryable=False) is explicitly non-retryable."""
    error = AIProviderError(retryable=False)
    assert error.retryable is False


def test_ai_config_error_not_retryable() -> None:
    """AIConfigError is never retryable (no fallback for config errors)."""
    error = AIConfigError()
    assert not hasattr(error, "retryable") or error.retryable is False
    assert isinstance(error, AIServiceError)


def test_ai_timeout_error_eligible_for_fallback() -> None:
    """AITimeoutError is a distinct exception type eligible for fallback."""
    error = AITimeoutError()
    assert isinstance(error, AIServiceError)
    assert error.status_code == 504


def test_ai_rate_limit_error_eligible_for_fallback() -> None:
    """AIRateLimitError is a distinct exception type eligible for fallback."""
    error = AIRateLimitError()
    assert isinstance(error, AIServiceError)
    assert error.status_code == 429
