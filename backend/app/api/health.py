"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["health"])


class HealthResponse(BaseModel):
    """Response returned by the health check."""

    status: str


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report whether the API process is running."""
    return HealthResponse(status="UP")

