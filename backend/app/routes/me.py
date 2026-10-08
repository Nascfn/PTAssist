"""GET /me: who the signed-in user is, and their role."""

from fastapi import APIRouter

from app.auth.current_user import CurrentUser
from app.schemas.me import MeResponse

router = APIRouter(tags=["me"])


@router.get("/me")
def get_me(user: CurrentUser) -> MeResponse:
    """Anyone signed in. Only ever returns the caller's own record."""
    return MeResponse(id=user.id, email=user.email, role=user.role)
