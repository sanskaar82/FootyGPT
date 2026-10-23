"""Pydantic schemas describing the data the API accepts and returns."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: Literal["healthy", "degraded"]
    application: str
    database: Literal["ok", "error"]
    ai_configured: bool


class WelcomeResponse(BaseModel):
    """Response body for ``GET /``."""

    message: str


class ChatRequest(BaseModel):
    """Body of a ``POST /chat`` request."""

    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: int | None = Field(default=None, gt=0)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        """Strip whitespace and reject messages that are only whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("Message must not be empty.")
        return value


class MessageOut(BaseModel):
    """One message in an API response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    """Body of a successful ``POST /chat`` response."""

    conversation_id: int
    user_message: MessageOut
    assistant_message: MessageOut
    model: str


class ConversationCreate(BaseModel):
    """Body of a ``POST /conversations`` request.

    ``title`` is optional; when omitted the conversation gets a default title.
    """

    title: str | None = Field(default=None, max_length=200)

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str | None) -> str | None:
        """Strip whitespace; treat a blank title as not supplied."""
        if value is None:
            return None
        value = value.strip()
        return value or None


class ConversationSummary(BaseModel):
    """One saved conversation, as shown in the sidebar."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationDetail(BaseModel):
    """A conversation together with all of its messages."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
