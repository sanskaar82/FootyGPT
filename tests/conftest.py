"""Shared pytest fixtures.

Tests use an in-memory SQLite database and a fake AI service, so the
development database is never touched and no API credits are used.
"""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend import models  # noqa: F401  (registers the tables with Base)
from backend.database import Base, create_db_engine, get_db
from backend.main import app
from backend.services.ai_service import get_ai_service


@pytest.fixture()
def anyio_backend() -> str:
    """Run ``@pytest.mark.anyio`` tests on asyncio only (trio is not installed)."""
    return "asyncio"


class FakeAIService:
    """Stand-in for ``AIService`` that records calls and returns fixed text."""

    model = "test/model"

    def __init__(self, answer: str = "A player is offside if they are nearer the goal line."):
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture()
def engine() -> Generator[Engine, None, None]:
    """A fresh in-memory database for each test."""
    test_engine = create_db_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture()
def session_factory(engine: Engine) -> sessionmaker:
    """Session factory bound to the test database."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def db_session(session_factory: sessionmaker) -> Generator[Session, None, None]:
    """A session bound to the test database."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def fake_ai() -> FakeAIService:
    """The fake AI service used by the test client."""
    return FakeAIService()


@pytest.fixture()
def client(
    session_factory: sessionmaker, fake_ai: FakeAIService
) -> Generator[TestClient, None, None]:
    """A FastAPI test client using the test database and fake AI service.

    The lifespan startup is not run here: it would create the development
    ``footygpt.db`` file. The test tables already exist instead.
    """

    def override_get_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ai_service] = lambda: fake_ai
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        test_client.close()
        app.dependency_overrides.clear()
