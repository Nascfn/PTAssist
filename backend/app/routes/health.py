"""GET /health: tells Docker and Azure that the API is up."""

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> HealthResponse:
    """Liveness check for Docker and Azure Container Apps.

    Public on purpose, because health probes send no token. It does not touch
    the database, so the API is not restarted when only the database is down.
    """
    return HealthResponse(status="ok")
