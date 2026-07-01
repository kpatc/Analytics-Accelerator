"""Integration tests for the /api/v1/health endpoint.

These tests spin up the FastAPI application in-process using the ASGI transport
from httpx — no real server or network is required.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    async def test_health_returns_200(self, async_client: AsyncClient) -> None:
        """Health check must respond with HTTP 200 OK."""
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200

    async def test_health_response_contains_status_ok(
        self, async_client: AsyncClient
    ) -> None:
        """Health check response body must contain status='ok'."""
        response = await async_client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "ok"

    async def test_health_response_contains_version(
        self, async_client: AsyncClient
    ) -> None:
        """Health check response must include a non-empty version string."""
        response = await async_client.get("/api/v1/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    async def test_health_response_contains_timestamp(
        self, async_client: AsyncClient
    ) -> None:
        """Health check response must include a UTC timestamp."""
        response = await async_client.get("/api/v1/health")
        data = response.json()
        assert "timestamp" in data
        assert data["timestamp"] is not None

    async def test_health_content_type_is_json(
        self, async_client: AsyncClient
    ) -> None:
        """Health check must return Content-Type: application/json."""
        response = await async_client.get("/api/v1/health")
        assert "application/json" in response.headers["content-type"]
