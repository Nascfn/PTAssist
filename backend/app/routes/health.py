"""GET /health: tells Docker and Azure that the API is up."""

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> HealthResponse:
    """Liveness probe: public on purpose (no token) and never touches the database."""
    return HealthResponse(status="ok")
