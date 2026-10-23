"""Routes for managing saved conversations."""

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from backend import crud
from backend.database import get_db
from backend.models import DEFAULT_CONVERSATION_TITLE
from backend.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationDetail, status_code=status.HTTP_201_CREATED)
def create_conversation(
    body: ConversationCreate | None = None, db: Session = Depends(get_db)
) -> ConversationDetail:
    """Create an empty conversation.

    The body is optional. Supply ``{"title": "..."}`` to name it; otherwise a
    default title is used and can be shown in the sidebar.
    """
    title = body.title if body and body.title else DEFAULT_CONVERSATION_TITLE
    conversation = crud.create_conversation(db, title)
    return ConversationDetail.model_validate(conversation)


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ConversationSummary]:
    """List saved conversations, newest first, without their messages."""
    summaries = crud.list_conversation_summaries(db, limit=limit, offset=offset)
    return [
        ConversationSummary(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=count,
        )
        for conversation, count in summaries
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: int, db: Session = Depends(get_db)
) -> ConversationDetail:
    """Return one conversation with all of its messages, oldest first."""
    conversation = crud.get_conversation_with_messages(db, conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation {conversation_id} not found."
        )
    return ConversationDetail.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)) -> Response:
    """Delete a conversation and all of its messages."""
    if not crud.delete_conversation(db, conversation_id):
        raise HTTPException(
            status_code=404, detail=f"Conversation {conversation_id} not found."
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
