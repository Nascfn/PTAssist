"""Database tables. Import every model here so Alembic autogenerate can see it."""

from app.models.base import Base
from app.models.user import Role, User

__all__ = ["Base", "Role", "User"]
