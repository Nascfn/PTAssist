"""Database tables. Import every model here so Alembic autogenerate can see it."""

from app.models.base import Base

__all__ = ["Base"]
