"""Entry point for the FootyGPT FastAPI application.

Run with: uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.database import init_db
from backend.exceptions import register_exception_handlers
from backend.routes import chat, conversations, health


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Prepare the database before the app starts serving requests."""
    init_db()
    yield


app = FastAPI(
    title="FootyGPT",
    description="Backend API for FootyGPT, an AI-powered football companion.",
    lifespan=lifespan,
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(conversations.router)
