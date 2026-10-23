"""Database engine, session factory, and table creation.

The backend uses SQLite through SQLAlchemy. This module owns the connection
setup so that routes and services never create sessions themselves.
"""

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    """Base class all SQLAlchemy models inherit from."""


def _enable_sqlite_foreign_keys(
    dbapi_connection: object, _connection_record: object
) -> None:
    """Turn on foreign-key enforcement for each new SQLite connection.

    SQLite ignores foreign keys unless this pragma is set per connection.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_engine(database_url: str, poolclass: type | None = None) -> Engine:
    """Build an engine for ``database_url``.

    Tests call this with an in-memory database URL (and ``StaticPool``) so
    the development database file is never touched.
    """
    connect_args = (
        {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    )
    engine = create_engine(
        database_url, connect_args=connect_args, poolclass=poolclass
    )
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


engine = create_db_engine(settings.database_url)

# Each request gets its own session from this factory.
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session per request.

    The session is always closed afterwards, even if the request fails. Any
    uncommitted work is rolled back first, so a failed request can never leave
    a half-written transaction behind.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create any missing tables.

    ``create_all`` only creates tables that do not exist yet, so existing
    data is never dropped or recreated.
    """
    from backend import models  # noqa: F401  (registers the tables)

    Base.metadata.create_all(bind=engine)
