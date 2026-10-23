"""Integration tests for POST /chat, with the AI service faked."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend import crud
from backend.exceptions import (
    AIConfigError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from backend.models import Message
from backend.services.prompts import SYSTEM_PROMPT
from tests.conftest import FakeAIService


def test_chat_creates_conversation_and_saves_exchange(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """A new question creates a conversation with the question and answer."""
    response = client.post("/chat", json={"message": "What is the offside rule?"})

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] > 0
    assert body["user_message"]["role"] == "user"
    assert body["user_message"]["content"] == "What is the offside rule?"
    assert body["assistant_message"]["role"] == "assistant"
    assert body["assistant_message"]["content"] == fake_ai.answer
    assert body["model"] == "test/model"

    conversation = crud.get_conversation(db_session, body["conversation_id"])
    assert conversation is not None
    assert conversation.title == "What is the offside rule?"
    assert [m.role for m in conversation.messages] == ["user", "assistant"]


def test_chat_sends_system_prompt_and_question(
    client: TestClient, fake_ai: FakeAIService
) -> None:
    """The AI receives the system prompt followed by the question."""
    client.post("/chat", json={"message": "What is a 4-3-3?"})

    sent = fake_ai.calls[0]
    assert sent[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert sent[-1] == {"role": "user", "content": "What is a 4-3-3?"}


def test_chat_continues_existing_conversation_with_history(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """A follow-up includes the earlier messages and appends to the conversation."""
    first = client.post("/chat", json={"message": "What is offside?"}).json()
    conversation_id = first["conversation_id"]

    second = client.post(
        "/chat", json={"message": "Can you give an example?", "conversation_id": conversation_id}
    )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id

    sent = fake_ai.calls[1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "What is offside?"
    assert sent[-1]["content"] == "Can you give an example?"

    conversation = crud.get_conversation(db_session, conversation_id)
    assert conversation is not None
    assert len(conversation.messages) == 4


def test_chat_history_is_limited(client: TestClient, fake_ai: FakeAIService) -> None:
    """Only the most recent history messages are sent to the AI."""
    from backend.config import settings

    conversation_id = client.post("/chat", json={"message": "Question 1"}).json()["conversation_id"]
    for i in range(2, 8):
        client.post(
            "/chat", json={"message": f"Question {i}", "conversation_id": conversation_id}
        )

    sent = fake_ai.calls[-1]
    # system + at most N history messages + the new question
    assert len(sent) == settings.ai_max_history_messages + 2
    # Six exchanges later there are 12 saved messages; only the newest 10 are
    # sent, so the oldest question and answer are dropped.
    assert sent[1]["content"] == "Question 2"
    assert all("Question 1" not in m["content"] for m in sent)
    assert sent[-1]["content"] == "Question 7"


def test_chat_unknown_conversation_returns_404(
    client: TestClient, fake_ai: FakeAIService
) -> None:
    """Continuing a missing conversation returns 404."""
    response = client.post("/chat", json={"message": "Hello", "conversation_id": 999})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    assert fake_ai.calls == []


def test_chat_rejects_blank_message(client: TestClient) -> None:
    """Whitespace-only messages are rejected with 422."""
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_rejects_missing_message(client: TestClient) -> None:
    """A request without a message is rejected with 422."""
    assert client.post("/chat", json={}).status_code == 422


def test_chat_rejects_message_too_long(client: TestClient) -> None:
    """Messages over 2,000 characters are rejected with 422."""
    assert client.post("/chat", json={"message": "x" * 2001}).status_code == 422


def test_chat_rejects_invalid_conversation_id(client: TestClient) -> None:
    """A non-positive conversation id is rejected with 422."""
    assert client.post("/chat", json={"message": "Hi", "conversation_id": 0}).status_code == 422
    assert client.post("/chat", json={"message": "Hi", "conversation_id": -1}).status_code == 422


def test_chat_auth_failure_returns_503_and_saves_nothing(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """A configuration error from the provider maps to 503 with no writes."""
    fake_ai.answer = AIConfigError()

    response = client.post("/chat", json={"message": "What is offside?"})

    assert response.status_code == 503
    assert db_session.query(Message).count() == 0


def test_chat_rate_limit_returns_429(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """Rate limiting maps to 429."""
    fake_ai.answer = AIRateLimitError()

    response = client.post("/chat", json={"message": "What is offside?"})

    assert response.status_code == 429
    assert db_session.query(Message).count() == 0


def test_chat_provider_error_returns_502(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """Provider errors map to 502."""
    fake_ai.answer = AIProviderError()

    response = client.post("/chat", json={"message": "What is offside?"})

    assert response.status_code == 502
    assert db_session.query(Message).count() == 0


def test_chat_timeout_returns_504(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """Timeouts map to 504."""
    fake_ai.answer = AITimeoutError()

    response = client.post("/chat", json={"message": "What is offside?"})

    assert response.status_code == 504
    assert db_session.query(Message).count() == 0


def test_chat_error_body_has_no_internals(
    client: TestClient, fake_ai: FakeAIService
) -> None:
    """Error responses use the standard detail shape, without stack traces."""
    fake_ai.answer = AIProviderError()

    body = client.post("/chat", json={"message": "What is offside?"}).json()

    assert set(body) == {"detail"}
    assert "Traceback" not in body["detail"]
