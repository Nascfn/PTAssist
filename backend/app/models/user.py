"""The users table: one row per person who has signed in with Clerk."""

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Role(StrEnum):
    PATIENT = "patient"
    THERAPIST = "therapist"


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('patient', 'therapist')", name="role"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    clerk_id: Mapped[str] = mapped_column(String(255), unique=True)
    email: Mapped[str] = mapped_column(String(320))
    # Lives here, not in Clerk. New users are patients; the team promotes therapists.
    role: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
