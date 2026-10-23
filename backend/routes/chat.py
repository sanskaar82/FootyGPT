"""The ``POST /chat`` route: ask the AI a football question."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import crud
from backend.config import settings
from backend.database import get_db
from backend.schemas import ChatRequest, ChatResponse
from backend.services.ai_service import AIService, get_ai_service
from backend.services.prompts import build_messages

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    ai_service: AIService = Depends(get_ai_service),
) -> ChatResponse:
    """Answer a football question and save the exchange.

    Creates a new conversation when ``conversation_id`` is omitted, otherwise
    continues the existing one. Nothing is saved if the AI call fails.
    """
    history: list[dict[str, str]] = []
    if request.conversation_id is not None:
        conversation = crud.get_conversation(db, request.conversation_id)
        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {request.conversation_id} not found.",
            )
        history = [
            {"role": message.role, "content": message.content}
            for message in crud.get_recent_messages(
                db, request.conversation_id, settings.ai_max_history_messages
            )
        ]

    messages = build_messages(request.message, history)

    # Any AI failure raises an AIServiceError, mapped to an HTTP status by
    # the exception handlers. The database is never written in that case.
    answer = await ai_service.complete(messages)

    # Report the model the AI service actually used, not a second lookup.
    model = ai_service.model or ""

    conversation, user_message, assistant_message = crud.save_exchange(
        db,
        request.conversation_id,
        question=request.message,
        answer=answer,
        model=model,
    )

    return ChatResponse(
        conversation_id=conversation.id,
        user_message=user_message,
        assistant_message=assistant_message,
        model=model,
    )
