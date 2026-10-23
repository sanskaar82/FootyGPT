"""Tests for the frontend's HTTP client, using mocked backend responses."""

import json

import httpx
import pytest

from frontend import api_client

BASE_URL = "https://test.backend.local"
CHAT_RESPONSE = {
    "conversation_id": 1,
    "user_message": {
        "id": 1,
        "role": "user",
        "content": "What is offside?",
        "created_at": "2026-10-23T14:30:00Z",
    },
    "assistant_message": {
        "id": 2,
        "role": "assistant",
        "content": "A player is offside if...",
        "created_at": "2026-10-23T14:30:04Z",
    },
    "model": "test/model",
}


def make_transport(handler) -> httpx.MockTransport:
    """Wrap a handler in a mock transport."""
    return httpx.MockTransport(handler)


def test_send_message_returns_backend_payload() -> None:
    """A successful call returns the backend's JSON unchanged."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_RESPONSE)

    result = api_client.send_chat_message(
        "What is offside?", base_url=BASE_URL, transport=make_transport(handler)
    )

    assert captured["url"] == f"{BASE_URL}/chat"
    assert captured["body"] == {"message": "What is offside?"}
    assert result == CHAT_RESPONSE
    assert result["assistant_message"]["content"] == "A player is offside if..."


def test_send_message_includes_conversation_id() -> None:
    """A follow-up sends the existing conversation id."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_RESPONSE)

    api_client.send_chat_message(
        "Example?", conversation_id=7, base_url=BASE_URL, transport=make_transport(handler)
    )

    assert captured["body"] == {"message": "Example?", "conversation_id": 7}


def test_send_message_omits_conversation_id_for_new_chat() -> None:
    """A new conversation sends only the message."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"message": "Hello"}
        return httpx.Response(200, json=CHAT_RESPONSE)

    api_client.send_chat_message("Hello", base_url=BASE_URL, transport=make_transport(handler))


@pytest.mark.parametrize(
    ("status", "expected_fragment"),
    [
        (404, "no longer exists"),
        (422, "couldn't be processed"),
        (429, "busy"),
        (502, "returned an error"),
        (503, "isn't fully set up"),
        (504, "too long"),
        (500, "Something went wrong"),
    ],
)
def test_error_statuses_become_friendly_messages(status: int, expected_fragment: str) -> None:
    """Unsuccessful responses raise APIError with a friendly message."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "internal detail"})

    with pytest.raises(api_client.APIError) as excinfo:
        api_client.send_chat_message("Hi", base_url=BASE_URL, transport=make_transport(handler))

    assert expected_fragment in excinfo.value.message
    assert "internal detail" not in excinfo.value.message


def test_connection_error_raises_backend_unavailable() -> None:
    """A connection failure is reported as the backend being unreachable."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(api_client.BackendUnavailableError) as excinfo:
        api_client.send_chat_message("Hi", base_url=BASE_URL, transport=make_transport(handler))

    assert "Can't reach" in excinfo.value.message


def test_timeout_raises_backend_unavailable() -> None:
    """A timeout is reported without crashing the page."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(api_client.BackendUnavailableError) as excinfo:
        api_client.send_chat_message("Hi", base_url=BASE_URL, transport=make_transport(handler))

    assert "taking too long" in excinfo.value.message


def test_invalid_json_raises_api_error() -> None:
    """A non-JSON success body is handled gracefully."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with pytest.raises(api_client.APIError):
        api_client.send_chat_message("Hi", base_url=BASE_URL, transport=make_transport(handler))


def test_health_ok() -> None:
    """check_health reports True for a healthy backend."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "status": "healthy",
                "application": "FootyGPT",
                "database": "ok",
                "ai_configured": True,
            },
        )

    assert api_client.check_health(base_url=BASE_URL, transport=make_transport(handler)) is True


def test_health_false_when_unreachable() -> None:
    """check_health reports False when the backend is down."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert api_client.check_health(base_url=BASE_URL, transport=make_transport(handler)) is False


def test_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """API_BASE_URL configures the backend location, trailing slash removed."""
    monkeypatch.setenv("API_BASE_URL", "http://example.test:9000/")
    assert api_client.get_base_url() == "http://example.test:9000"


def test_base_url_defaults_to_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without configuration, local development defaults apply."""
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("BACKEND_URL", raising=False)
    assert api_client.get_base_url() == "http://localhost:8000"


# --- Conversation history calls (Phase 6) ---

CONVERSATION_DETAIL = {
    "id": 3,
    "title": "Offside",
    "created_at": "2026-10-23T14:30:00Z",
    "updated_at": "2026-10-23T14:32:00Z",
    "messages": [
        {"id": 1, "role": "user", "content": "What is offside?", "created_at": "2026-10-23T14:30:00Z"},
        {"id": 2, "role": "assistant", "content": "It is...", "created_at": "2026-10-23T14:30:04Z"},
    ],
}


def test_create_conversation_posts_title() -> None:
    """create_conversation posts the title and returns the new conversation."""
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={**CONVERSATION_DETAIL, "messages": []})

    result = api_client.create_conversation(
        "Offside", base_url=BASE_URL, transport=make_transport(handler)
    )

    assert captured == {"method": "POST", "path": "/conversations", "body": {"title": "Offside"}}
    assert result["id"] == 3


def test_create_conversation_without_title_sends_empty_body() -> None:
    """An untitled conversation sends an empty JSON object."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json=CONVERSATION_DETAIL)

    api_client.create_conversation(base_url=BASE_URL, transport=make_transport(handler))

    assert captured["body"] == {}


def test_list_conversations_returns_summaries() -> None:
    """list_conversations reads GET /conversations."""
    summaries = [
        {"id": 3, "title": "Offside", "created_at": "a", "updated_at": "b", "message_count": 2}
    ]
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["limit"] = request.url.params.get("limit")
        return httpx.Response(200, json=summaries)

    result = api_client.list_conversations(base_url=BASE_URL, transport=make_transport(handler))

    assert captured["path"] == "/conversations"
    assert captured["limit"] == "50"
    assert result == summaries


def test_get_conversation_returns_detail() -> None:
    """get_conversation reads one conversation with its messages."""
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return httpx.Response(200, json=CONVERSATION_DETAIL)

    result = api_client.get_conversation(3, base_url=BASE_URL, transport=make_transport(handler))

    assert captured["path"] == "/conversations/3"
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]


def test_delete_conversation_sends_delete() -> None:
    """delete_conversation calls DELETE and accepts an empty 204 body."""
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(204)

    api_client.delete_conversation(3, base_url=BASE_URL, transport=make_transport(handler))

    assert captured == {"method": "DELETE", "path": "/conversations/3"}


def test_missing_conversation_is_friendly() -> None:
    """A 404 from a conversation call becomes a friendly message."""

    def handler(request):
        return httpx.Response(404, json={"detail": "Conversation 3 not found."})

    with pytest.raises(api_client.APIError) as excinfo:
        api_client.get_conversation(3, base_url=BASE_URL, transport=make_transport(handler))

    assert "no longer exists" in excinfo.value.message


def test_conversation_calls_report_backend_down() -> None:
    """Network failures from any conversation call are reported clearly."""

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    for call in (
        lambda: api_client.list_conversations(base_url=BASE_URL, transport=make_transport(handler)),
        lambda: api_client.get_conversation(1, base_url=BASE_URL, transport=make_transport(handler)),
        lambda: api_client.delete_conversation(1, base_url=BASE_URL, transport=make_transport(handler)),
    ):
        with pytest.raises(api_client.BackendUnavailableError):
            call()


def test_frontend_never_needs_a_backend_secret() -> None:
    """The frontend sends no authorization header; the key stays server-side."""
    captured = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=CHAT_RESPONSE)

    api_client.send_chat_message("Hi", base_url=BASE_URL, transport=make_transport(handler))

    assert "authorization" not in {k.lower() for k in captured["headers"]}
    assert not any("openrouter" in v.lower() for v in captured["headers"].values())


def test_backend_status_codes_stay_friendly_for_every_call() -> None:
    """A 500 from any endpoint shows a generic message, not server internals."""

    def handler(request):
        return httpx.Response(500, json={"detail": "Traceback ... secret"})

    with pytest.raises(api_client.APIError) as excinfo:
        api_client.list_conversations(base_url=BASE_URL, transport=make_transport(handler))

    assert "Traceback" not in excinfo.value.message
    assert "secret" not in excinfo.value.message


def test_dotenv_is_loaded_for_streamlit(tmp_path) -> None:
    """API_BASE_URL in .env is picked up without exporting it (Streamlit does
    not read .env itself). Run in a subprocess with a temporary ENV_FILE."""
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    env_file = tmp_path / "test.env"
    env_file.write_text("API_BASE_URL=http://from-dotenv:8123\n", encoding="utf-8")

    script = (
        "import os;"
        "os.environ.pop('API_BASE_URL', None);"
        "os.environ.pop('BACKEND_URL', None);"
        "import frontend.api_client as c; print(c.get_base_url())"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "ENV_FILE": str(env_file), "PYTHONPATH": str(project_root)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "http://from-dotenv:8123"
