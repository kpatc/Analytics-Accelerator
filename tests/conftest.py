"""Pytest configuration and shared fixtures.

Fixtures defined here are available to all test modules without import.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.bcgx.config.settings import Settings, get_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Return a Settings instance with test-friendly overrides.

    We construct Settings directly (bypassing lru_cache) so each test session
    gets a predictable, isolated configuration.
    """
    return Settings(
        app_env="testing",
        app_debug=True,
        log_level="DEBUG",
        random_seed=42,
    )


@pytest_asyncio.fixture()
async def async_client() -> AsyncClient:  # type: ignore[misc]
    """Return an httpx.AsyncClient wired to the FastAPI app.

    Uses the ASGI transport so no real network socket is opened.  Suitable
    for integration tests that need to exercise the full request/response cycle.
    """
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client
