"""Tests for the database layer: models, relationships, and sessions."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend import crud
from backend.models import Conversation, Message


def test_create_conversation(db_session: Session) -> None:
    """A conversation can be created and read back."""
    conversation = crud.create_conversation(db_session, "What is the offside rule?")

    assert conversation.id is not None
    assert conversation.title == "What is the offside rule?"
    assert conversation.created_at is not None
    assert conversation.updated_at is not None
    assert crud.get_conversation(db_session, conversation.id) is not None


def test_add_message_to_conversation(db_session: Session) -> None:
    """A message can be associated with a conversation."""
    conversation = crud.create_conversation(db_session, "Tactics")
    message = crud.add_message(
        db_session, conversation.id, role="user", content="What is a high press?"
    )

    assert message.id is not None
    assert message.conversation_id == conversation.id
    assert message.role == "user"
    assert message.content == "What is a high press?"
    assert message.model is None


def test_conversation_has_multiple_messages_in_order(db_session: Session) -> None:
    """A conversation can hold several messages, returned in insertion order."""
    conversation = crud.create_conversation(db_session, "Formations")
    crud.add_message(db_session, conversation.id, "user", "Explain 4-3-3.")
    crud.add_message(
        db_session, conversation.id, "assistant", "It uses three midfielders.", model="test-model"
    )

    db_session.expire_all()
    loaded = crud.get_conversation(db_session, conversation.id)
    assert loaded is not None
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert [m.content for m in loaded.messages] == [
        "Explain 4-3-3.",
        "It uses three midfielders.",
    ]


def test_get_conversation_missing_returns_none(db_session: Session) -> None:
    """Looking up an unknown conversation returns None."""
    assert crud.get_conversation(db_session, 999) is None


def test_add_message_to_unknown_conversation_raises(db_session: Session) -> None:
    """Adding a message to a missing conversation is rejected."""
    with pytest.raises(ValueError):
        crud.add_message(db_session, 999, "user", "Hello?")


def test_invalid_role_is_rejected(db_session: Session) -> None:
    """The role check constraint only allows 'user' and 'assistant'."""
    conversation = crud.create_conversation(db_session, "Roles")
    with pytest.raises(IntegrityError):
        crud.add_message(db_session, conversation.id, "system", "Not allowed")
    db_session.rollback()


def test_foreign_key_is_enforced(db_session: Session) -> None:
    """SQLite foreign-key enforcement is enabled on the test engine."""
    enabled = db_session.execute(text("PRAGMA foreign_keys")).scalar()
    assert enabled == 1


def test_delete_conversation_cascades_to_messages(db_session: Session) -> None:
    """Deleting a conversation removes its messages."""
    conversation = crud.create_conversation(db_session, "History")
    crud.add_message(db_session, conversation.id, "user", "First")
    crud.add_message(db_session, conversation.id, "assistant", "Second", model="m")
    conversation_id = conversation.id

    assert crud.delete_conversation(db_session, conversation_id) is True
    assert crud.get_conversation(db_session, conversation_id) is None
    assert db_session.query(Message).filter_by(conversation_id=conversation_id).count() == 0


def test_delete_missing_conversation_returns_false(db_session: Session) -> None:
    """Deleting an unknown conversation reports that nothing was deleted."""
    assert crud.delete_conversation(db_session, 999) is False


def test_list_conversations_newest_first(db_session: Session) -> None:
    """Conversations are listed newest first."""
    first = crud.create_conversation(db_session, "First")
    second = crud.create_conversation(db_session, "Second")

    listed = crud.list_conversations(db_session)
    assert [c.title for c in listed] == [second.title, first.title]


def test_sessions_are_independent(session_factory: sessionmaker) -> None:
    """Each session sees committed data, and closing one leaves it intact."""
    session_a = session_factory()
    conversation = crud.create_conversation(session_a, "Shared")
    session_a.close()

    session_b = session_factory()
    try:
        assert crud.get_conversation(session_b, conversation.id) is not None
    finally:
        session_b.close()


def test_failed_transaction_can_be_rolled_back(db_session: Session) -> None:
    """An error inside a session does not leave a partial transaction open."""
    conversation = crud.create_conversation(db_session, "Rollback")
    db_session.add(Message(conversation_id=conversation.id, role="invalid", content="x"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    assert db_session.query(Message).count() == 0


def test_get_db_rolls_back_uncommitted_work_on_error(monkeypatch) -> None:
    """A failing request neither commits nor leaves a session dangling."""
    from backend import database

    closed = []

    class SpySession:
        rolled_back = False
        committed = False

        def rollback(self) -> None:
            self.rolled_back = True

        def close(self) -> None:
            closed.append(True)

    spy = SpySession()
    monkeypatch.setattr(database, "SessionLocal", lambda: spy)

    generator = database.get_db()
    next(generator)
    with pytest.raises(RuntimeError):
        generator.throw(RuntimeError("boom"))

    assert spy.rolled_back is True
    assert closed == [True]


def test_get_db_closes_session_on_success(monkeypatch) -> None:
    """A successful request closes its session."""
    from backend import database

    closed = []

    class SpySession:
        def rollback(self) -> None:  # pragma: no cover
            raise AssertionError("should not roll back")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(database, "SessionLocal", lambda: SpySession())

    generator = database.get_db()
    next(generator)
    generator.close()

    assert closed == [True]
