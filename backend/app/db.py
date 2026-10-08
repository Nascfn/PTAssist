"""Database connection: the SQLAlchemy engine and a session per request for routes."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import DATABASE_URL

# hide_parameters keeps query values (like emails) out of logs and error messages.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, hide_parameters=True)


def get_db() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
