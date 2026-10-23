"""Routes for checking that the backend is running."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.schemas import HealthResponse, WelcomeResponse

router = APIRouter()


@router.get("/", response_model=WelcomeResponse)
def read_root() -> WelcomeResponse:
    """Return a simple welcome message."""
    return WelcomeResponse(message="Welcome to FootyGPT, your AI football companion.")


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Confirm the backend is up and report database and AI configuration.

    This never calls OpenRouter, so it costs nothing and adds no latency.
    """
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError:
        database = "error"

    return HealthResponse(
        status="healthy" if database == "ok" else "degraded",
        application="FootyGPT",
        database=database,
        ai_configured=settings.ai_configured,
    )
