"""Helpers for calling the FootyGPT backend.

The frontend never talks to OpenRouter or the database directly. Every request
goes through the FastAPI backend, and the API key stays server-side.
"""

import os
from pathlib import Path
from typing import Any

import httpx

# The project's .env file. ENV_FILE overrides it, which tests use so they
# never depend on a developer's real .env.
ENV_FILE = os.getenv("ENV_FILE") or str(
    Path(__file__).resolve().parent.parent / ".env"
)
try:
    from dotenv import load_dotenv

    # The backend reads .env through pydantic-settings; Streamlit does not, so
    # load it here too or API_BASE_URL from .env would be silently ignored.
    load_dotenv(ENV_FILE)
except ImportError:  # python-dotenv is optional for the frontend
    pass

DEFAULT_BASE_URL = "http://localhost:8000"

# FastAPI can wait up to AI_TIMEOUT_SECONDS (30 by default) for the model, so
# the read timeout here must be longer than that.
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 60.0


class APIError(Exception):
    """An error with a message that is safe to show to the user."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BackendUnavailableError(APIError):
    """The backend could not be reached at all."""


def get_base_url() -> str:
    """Return the backend URL from the environment, or the local default."""
    return (
        os.getenv("API_BASE_URL")
        or os.getenv("BACKEND_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _make_client(base_url: str, transport: httpx.BaseTransport | None) -> httpx.Client:
    """Build an HTTP client with sensible timeouts."""
    return httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
        transport=transport,
    )


def _friendly_message(status_code: int) -> str:
    """Translate a backend status code into a friendly message."""
    return {
        404: "That conversation no longer exists.",
        422: "Your question couldn't be processed. Please check it and try again.",
        429: "FootyGPT is busy. Please wait a moment and try again.",
        502: "The AI service returned an error. Please try again.",
        503: "FootyGPT isn't fully set up yet. Please try again later.",
        504: "The AI took too long to respond. Please try again.",
    }.get(status_code, "Something went wrong. Please try again.")


def _request(
    method: str,
    path: str,
    base_url: str | None,
    transport: httpx.BaseTransport | None,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """Send one request, raising friendly errors for network and HTTP failures.

    Shared by every call below so error handling stays in one place.
    """
    try:
        with _make_client(base_url or get_base_url(), transport) as client:
            response = client.request(method, path, json=json, params=params)
    except httpx.TimeoutException as exc:
        raise BackendUnavailableError(
            "FootyGPT is taking too long to respond. Please try again."
        ) from exc
    except httpx.HTTPError as exc:
        raise BackendUnavailableError(
            "Can't reach the FootyGPT server. Is the backend running?"
        ) from exc

    if response.status_code >= 400:
        raise APIError(_friendly_message(response.status_code))
    return response


def _json(response: httpx.Response) -> Any:
    """Parse a JSON body, converting malformed responses into a friendly error."""
    try:
        return response.json()
    except ValueError as exc:
        raise APIError("Something went wrong. Please try again.") from exc


def send_chat_message(
    message: str,
    conversation_id: int | None = None,
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Send a question to ``POST /chat`` and return the parsed response.

    Returns the backend's JSON body, which contains ``conversation_id``,
    ``user_message``, ``assistant_message`` and ``model``.
    """
    payload: dict[str, Any] = {"message": message}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id

    response = _request("POST", "/chat", base_url, transport, json=payload)
    return _json(response)


def create_conversation(
    title: str | None = None,
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Create an empty conversation via ``POST /conversations``."""
    payload: dict[str, Any] = {}
    if title:
        payload["title"] = title

    response = _request("POST", "/conversations", base_url, transport, json=payload)
    return _json(response)


def list_conversations(
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return saved conversations from ``GET /conversations``, newest first."""
    response = _request(
        "GET", "/conversations", base_url, transport, params={"limit": limit}
    )
    return _json(response)


def get_conversation(
    conversation_id: int,
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Return a conversation and its messages via ``GET /conversations/{id}``."""
    response = _request(
        "GET", f"/conversations/{conversation_id}", base_url, transport
    )
    return _json(response)


def delete_conversation(
    conversation_id: int,
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Delete a conversation via ``DELETE /conversations/{id}``."""
    _request("DELETE", f"/conversations/{conversation_id}", base_url, transport)


def check_health(
    base_url: str | None = None, transport: httpx.BaseTransport | None = None
) -> bool:
    """Return ``True`` if ``GET /health`` reports a healthy backend."""
    try:
        with _make_client(base_url or get_base_url(), transport) as client:
            response = client.get("/health", timeout=CONNECT_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return False

    if response.status_code != 200:
        return False
    try:
        return response.json().get("status") == "healthy"
    except ValueError:
        return False
