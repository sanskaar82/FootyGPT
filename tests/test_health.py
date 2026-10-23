"""Tests for the health and welcome endpoints."""

from fastapi.testclient import TestClient


def test_health_returns_expected_body(client: TestClient) -> None:
    """GET /health returns 200 with the documented body."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["application"] == "FootyGPT"
    assert body["database"] == "ok"
    assert isinstance(body["ai_configured"], bool)


def test_root_returns_welcome_message(client: TestClient) -> None:
    """GET / returns 200 with a welcome message."""
    response = client.get("/")

    assert response.status_code == 200
    assert "FootyGPT" in response.json()["message"]
