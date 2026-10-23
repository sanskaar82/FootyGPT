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
    AIProviderError,
    DATABASE_ERROR_MESSAGE,
    GENERIC_ERROR_MESSAGE,
)
from backend.main import app
from backend.models import Message
from backend.services.ai_service import MAX_ANSWER_CHARS, AIService, get_ai_service

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
