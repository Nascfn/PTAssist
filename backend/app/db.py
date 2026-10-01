"""Database connection: the SQLAlchemy engine and a session per request for routes."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one database session per request."""
    with SessionLocal() as session:
        yield session
