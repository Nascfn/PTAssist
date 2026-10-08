"""Response model for GET /me."""

import uuid

from pydantic import BaseModel

from app.models.user import Role


class MeResponse(BaseModel):
    id: uuid.UUID
    email: str
    role: Role
