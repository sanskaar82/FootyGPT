"""Error classes and the handlers that turn them into HTTP responses.

Keeping these separate from HTTP concerns means the AI service and the
database layer never need to know about FastAPI. Every error is translated
into a response in one place, so users always see the same shape:

    {"detail": "A friendly, non-technical message."}

Details that could help an attacker (stack traces, SQL, API keys, provider
response bodies) are only written to the server log, never returned.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

GENERIC_ERROR_MESSAGE = "Something unexpected happened. Please try again."
DATABASE_ERROR_MESSAGE = (
    "Something went wrong saving your conversation. Please try again."
)


class AIServiceError(Exception):
    """Base class for problems talking to the AI provider."""

    status_code = 502
    message = "The AI service returned an error. Please try again."


class AIConfigError(AIServiceError):
    """The API key or model is missing, invalid, or rejected."""

    status_code = 503
    message = "AI service is not configured."


class AIRateLimitError(AIServiceError):
    """The provider reported too many requests."""

    status_code = 429
    message = "The AI service is busy. Please wait a moment and try again."


class AITimeoutError(AIServiceError):
    """The provider did not answer within the configured timeout."""

    status_code = 504
    message = "The AI took too long to respond. Please try again."


class AIProviderError(AIServiceError):
    """The provider returned an error, or an empty/invalid response."""

    status_code = 502
    message = "The AI service returned an error. Please try again."


def _ai_error_response(request: Request, error: AIServiceError) -> JSONResponse:
    """Log the error type (never the key or message content) and reply."""
    logger.warning(
        "AI request failed on %s %s: %s",
        request.method,
        request.url.path,
        type(error).__name__,
    )
    return JSONResponse(status_code=error.status_code, content={"detail": error.message})


def _database_error_response(request: Request, error: SQLAlchemyError) -> JSONResponse:
    """Report a database failure without exposing SQL or schema details."""
    logger.error(
        "Database error on %s %s: %s",
        request.method,
        request.url.path,
        type(error).__name__,
    )
    return JSONResponse(status_code=503, content={"detail": DATABASE_ERROR_MESSAGE})


def _unexpected_error_response(request: Request, error: Exception) -> JSONResponse:
    """Last line of defence: log the exception, return a generic message.

    The log line stores the exception type and the request path only, so
    stack traces stay in the server log rather than the response.
    """
    logger.exception(
        "Unexpected error on %s %s: %s", request.method, request.url.path, type(error).__name__
    )
    return JSONResponse(status_code=500, content={"detail": GENERIC_ERROR_MESSAGE})


def register_exception_handlers(app: FastAPI) -> None:
    """Register every error mapping in one place."""
    for error_class in (
        AIServiceError,
        AIConfigError,
        AIRateLimitError,
        AITimeoutError,
        AIProviderError,
    ):
        app.add_exception_handler(error_class, _ai_error_response)
    app.add_exception_handler(SQLAlchemyError, _database_error_response)
    app.add_exception_handler(Exception, _unexpected_error_response)
