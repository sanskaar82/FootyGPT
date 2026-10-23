"""Integration tests for the conversation endpoints and chat persistence."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.exceptions import AIProviderError, AIRateLimitError
from backend.models import Conversation, Message
from tests.conftest import FakeAIService


def _create_conversation(client: TestClient, title: str | None = None) -> dict:
    body = {"title": title} if title else {}
    response = client.post("/conversations", json=body)
    assert response.status_code == 201
    return response.json()


def test_create_conversation(client: TestClient, db_session: Session) -> None:
    """POST /conversations creates an empty conversation."""
    body = _create_conversation(client, "Offside explained")

    assert body["id"] > 0
    assert body["title"] == "Offside explained"
    assert body["messages"] == []
    assert db_session.get(Conversation, body["id"]) is not None


def test_create_conversation_without_title_uses_default(client: TestClient) -> None:
    """An empty body is allowed and gets a default title."""
    response = client.post("/conversations", json={})

    assert response.status_code == 201
    assert response.json()["title"] == "New conversation"


def test_create_conversation_with_no_body(client: TestClient) -> None:
    """The request body is optional."""
    response = client.post("/conversations")

    assert response.status_code == 201


def test_create_conversation_rejects_blank_title(client: TestClient) -> None:
    """A whitespace-only title is treated as not supplied."""
    response = client.post("/conversations", json={"title": "   "})

    assert response.status_code == 201
    assert response.json()["title"] == "New conversation"


def test_create_conversation_rejects_long_title(client: TestClient) -> None:
    """Titles over 200 characters are rejected."""
    response = client.post("/conversations", json={"title": "x" * 201})

    assert response.status_code == 422


def test_list_conversations_newest_first(client: TestClient) -> None:
    """GET /conversations lists conversations newest first with message counts."""
    first = _create_conversation(client, "First")
    client.post("/chat", json={"message": "Question", "conversation_id": first["id"]})
    second = _create_conversation(client, "Second")

    listed = client.get("/conversations").json()

    assert [c["title"] for c in listed] == ["Second", "First"]
    assert listed[0]["message_count"] == 0
    assert listed[1]["message_count"] == 2
    assert listed[1]["id"] == first["id"]


def test_list_conversations_empty(client: TestClient) -> None:
    """An empty database returns an empty list."""
    response = client.get("/conversations")

    assert response.status_code == 200
    assert response.json() == []


def test_list_conversations_pagination(client: TestClient) -> None:
    """limit and offset page through the list."""
    for i in range(3):
        _create_conversation(client, f"C{i}")

    page = client.get("/conversations", params={"limit": 1, "offset": 1}).json()

    assert len(page) == 1
    assert page[0]["title"] == "C1"


def test_list_conversations_rejects_bad_pagination(client: TestClient) -> None:
    """Invalid paging values return 422."""
    assert client.get("/conversations", params={"limit": 0}).status_code == 422
    assert client.get("/conversations", params={"offset": -1}).status_code == 422


def test_get_conversation_with_messages(client: TestClient) -> None:
    """GET /conversations/{id} returns messages oldest first."""
    conversation = _create_conversation(client, "Formations")
    client.post("/chat", json={"message": "Explain 4-3-3.", "conversation_id": conversation["id"]})

    body = client.get(f"/conversations/{conversation['id']}").json()

    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "Explain 4-3-3."


def test_get_missing_conversation_returns_404(client: TestClient) -> None:
    """A nonexistent conversation id returns 404."""
    response = client.get("/conversations/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation 999 not found."}


def test_get_conversation_rejects_non_integer_id(client: TestClient) -> None:
    """A non-integer id returns 422."""
    assert client.get("/conversations/abc").status_code == 422


def test_delete_conversation_removes_messages(
    client: TestClient, db_session: Session
) -> None:
    """DELETE removes the conversation and its messages."""
    conversation = _create_conversation(client, "Delete me")
    client.post("/chat", json={"message": "Hello", "conversation_id": conversation["id"]})

    response = client.delete(f"/conversations/{conversation['id']}")

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/conversations/{conversation['id']}").status_code == 404
    assert db_session.get(Conversation, conversation["id"]) is None
    assert (
        db_session.query(Message)
        .filter_by(conversation_id=conversation["id"])
        .count()
        == 0
    )


def test_delete_missing_conversation_returns_404(client: TestClient) -> None:
    """Deleting an unknown conversation returns 404."""
    assert client.delete("/conversations/999").status_code == 404


def test_delete_twice_second_call_404(client: TestClient) -> None:
    """A second delete of the same conversation returns 404."""
    conversation = _create_conversation(client, "Twice")

    assert client.delete(f"/conversations/{conversation['id']}").status_code == 204
    assert client.delete(f"/conversations/{conversation['id']}").status_code == 404


def test_chat_saves_user_and_assistant_messages(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """A chat exchange saves both messages against the right conversation."""
    conversation = _create_conversation(client, "Chat")

    body = client.post(
        "/chat", json={"message": "What is a high press?", "conversation_id": conversation["id"]}
    ).json()

    assert body["conversation_id"] == conversation["id"]
    saved = (
        db_session.query(Message)
        .filter_by(conversation_id=conversation["id"])
        .order_by(Message.id)
        .all()
    )
    assert [(m.role, m.content) for m in saved] == [
        ("user", "What is a high press?"),
        ("assistant", fake_ai.answer),
    ]
    assert saved[1].model == "test/model"


def test_chat_titles_empty_conversation_from_first_question(client: TestClient) -> None:
    """An untitled conversation is named after its first question."""
    conversation = _create_conversation(client)

    client.post("/chat", json={"message": "Who won the 1966 World Cup?", "conversation_id": conversation["id"]})

    body = client.get(f"/conversations/{conversation['id']}").json()
    assert body["title"] == "Who won the 1966 World Cup?"


def test_chat_keeps_user_supplied_title(client: TestClient) -> None:
    """A conversation the user named keeps that name after its first question."""
    conversation = _create_conversation(client, "My research")

    client.post("/chat", json={"message": "What is a false nine?", "conversation_id": conversation["id"]})

    body = client.get(f"/conversations/{conversation['id']}").json()
    assert body["title"] == "My research"


def test_chat_does_not_rename_existing_conversation(client: TestClient) -> None:
    """Later questions do not change the title."""
    conversation = _create_conversation(client, "Keep me")
    client.post("/chat", json={"message": "Question one", "conversation_id": conversation["id"]})
    client.post("/chat", json={"message": "Question two", "conversation_id": conversation["id"]})

    body = client.get(f"/conversations/{conversation['id']}").json()
    assert body["title"] == "Keep me"
    assert len(body["messages"]) == 4


def test_chat_continues_conversation_across_requests(
    client: TestClient, fake_ai: FakeAIService
) -> None:
    """A follow-up in the same conversation appends rather than duplicating."""
    conversation = _create_conversation(client, "Continue")

    client.post("/chat", json={"message": "First", "conversation_id": conversation["id"]})
    client.post("/chat", json={"message": "Second", "conversation_id": conversation["id"]})

    messages = client.get(f"/conversations/{conversation['id']}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == [
        "First",
        fake_ai.answer,
        "Second",
        fake_ai.answer,
    ]


def test_chat_unknown_conversation_returns_404_and_saves_nothing(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """Chatting into a missing conversation returns 404 with no writes."""
    response = client.post("/chat", json={"message": "Hi", "conversation_id": 999})

    assert response.status_code == 404
    assert fake_ai.calls == []
    assert db_session.query(Message).count() == 0


def test_ai_failure_saves_nothing_in_existing_conversation(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """A failed AI call leaves the conversation and message count unchanged."""
    conversation = _create_conversation(client, "Failures")
    client.post("/chat", json={"message": "First", "conversation_id": conversation["id"]})
    before = db_session.query(Message).count()

    fake_ai.answer = AIProviderError()
    response = client.post(
        "/chat", json={"message": "Second", "conversation_id": conversation["id"]}
    )

    assert response.status_code == 502
    assert db_session.query(Message).count() == before


def test_rate_limit_saves_nothing_in_existing_conversation(
    client: TestClient, db_session: Session, fake_ai: FakeAIService
) -> None:
    """A rate-limited AI call also writes nothing."""
    conversation = _create_conversation(client, "Rate limited")
    fake_ai.answer = AIRateLimitError()

    response = client.post(
        "/chat", json={"message": "Hi", "conversation_id": conversation["id"]}
    )

    assert response.status_code == 429
    assert db_session.query(Message).count() == 0


def test_conversation_survives_new_session(
    client: TestClient, session_factory
) -> None:
    """Saved conversations are readable from a fresh session (a restart)."""
    from backend import crud

    conversation = _create_conversation(client, "Persistent")
    client.post("/chat", json={"message": "Remember me", "conversation_id": conversation["id"]})

    other = session_factory()
    try:
        loaded = crud.get_conversation_with_messages(other, conversation["id"])
        assert loaded is not None
        assert [m.content for m in loaded.messages][0] == "Remember me"
    finally:
        other.close()
