"""Database operations for conversations and messages.

Keeping queries here (rather than in route functions) makes them reusable
and easy to test.
"""

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models import DEFAULT_CONVERSATION_TITLE, Conversation, Message


def create_conversation(db: Session, title: str) -> Conversation:
    """Create and save a new conversation, returning it."""
    conversation = Conversation(title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def add_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str,
    model: str | None = None,
) -> Message:
    """Add a message to an existing conversation and update its timestamp."""
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found.")

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        model=model,
    )
    conversation.updated_at = datetime.now(timezone.utc)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_recent_messages(
    db: Session, conversation_id: int, limit: int
) -> list[Message]:
    """Return the most recent ``limit`` messages, oldest first."""
    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(limit)
    )
    return list(reversed(list(db.scalars(statement))))


def save_exchange(
    db: Session,
    conversation_id: int | None,
    question: str,
    answer: str,
    model: str,
) -> tuple[Conversation, Message, Message]:
    """Save a question and its answer together in one transaction.

    A new conversation is created when ``conversation_id`` is ``None``. Nothing
    is written if the commit fails. Returns the conversation, the user message,
    and the assistant message.
    """
    is_first_exchange = True
    if conversation_id is None:
        conversation = Conversation(title=question[:200])
        db.add(conversation)
        db.flush()  # assign an id without committing yet
    else:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found.")
        is_first_exchange = not conversation.messages

    # An empty conversation created via POST /conversations without a title
    # gets one from its first question, matching how /chat names a new chat.
    if is_first_exchange and conversation.title == DEFAULT_CONVERSATION_TITLE:
        conversation.title = question[:200]

    now = datetime.now(timezone.utc)
    user_message = Message(
        conversation_id=conversation.id, role="user", content=question
    )
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        model=model,
    )
    conversation.updated_at = now
    db.add_all([user_message, assistant_message])
    try:
        db.commit()
    except Exception:
        # Leave the database exactly as it was before this exchange.
        db.rollback()
        raise
    db.refresh(user_message)
    db.refresh(assistant_message)
    return conversation, user_message, assistant_message


def get_conversation(db: Session, conversation_id: int) -> Conversation | None:
    """Return one conversation, or ``None`` if it does not exist."""
    return db.get(Conversation, conversation_id)


def list_conversations(db: Session) -> list[Conversation]:
    """Return all conversations, newest first."""
    statement = select(Conversation).order_by(Conversation.updated_at.desc())
    return list(db.scalars(statement))


def list_conversation_summaries(
    db: Session, limit: int = 50, offset: int = 0
) -> list[tuple[Conversation, int]]:
    """Return conversations with their message counts, newest first.

    Each item is a ``(conversation, message_count)`` pair. ``limit`` and
    ``offset`` implement simple paging for the sidebar.
    """
    statement = (
        select(Conversation, func.count(Message.id))
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return [(conversation, count) for conversation, count in db.execute(statement)]


def get_conversation_with_messages(
    db: Session, conversation_id: int
) -> Conversation | None:
    """Return one conversation with its messages loaded, oldest first."""
    conversation = db.get(Conversation, conversation_id)
    if conversation is not None:
        # Load the relationship now, while the session is open.
        _ = conversation.messages
    return conversation


def delete_conversation(db: Session, conversation_id: int) -> bool:
    """Delete a conversation and its messages.

    Messages are removed by the relationship's ``cascade="all, delete-orphan"``
    and by the database's ``ON DELETE CASCADE``. Returns ``True`` if the
    conversation existed, ``False`` otherwise.
    """
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        return False
    db.delete(conversation)
    db.commit()
    return True
