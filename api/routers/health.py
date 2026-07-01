"""Health check router."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    """Schema for the health check response."""

    status: str
    version: str
    timestamp: datetime


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Return service health status, version, and current UTC timestamp."""
    return HealthResponse(
        status="ok",
        version=_VERSION,
        timestamp=datetime.now(tz=timezone.utc),
    )
