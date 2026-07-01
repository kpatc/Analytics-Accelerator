"""BCG X Analytics Accelerator — FastAPI application factory.

This module defines the FastAPI app and wires up:
- Lifespan context (startup / shutdown hooks)
- CORS middleware
- All versioned API routers
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import copilot, data, health, models, recommendations, simulation
from src.bcgx.config.logging import setup_logging
from src.bcgx.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager.

    Handles startup and shutdown logic:
    - Startup: configure structured logging, validate settings.
    - Shutdown: flush any remaining log records.
    """
    settings = get_settings()
    setup_logging(level=settings.log_level)

    from loguru import logger

    logger.info(
        "BCG X Analytics API starting — env={env} debug={debug}",
        env=settings.app_env,
        debug=settings.app_debug,
    )

    yield  # ← application runs here

    logger.info("BCG X Analytics API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="BCG X NovaMart Analytics API",
        description=(
            "Production analytics API for the BCG X / NovaMart retail consulting "
            "engagement. Exposes data summaries, ML model predictions, scenario "
            "simulations, strategic recommendations, and an AI Copilot."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = ["*"] if settings.app_debug else [settings.dashboard.api_url]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    PREFIX = "/api/v1"

    app.include_router(health.router, prefix=PREFIX, tags=["Health"])
    app.include_router(data.router, prefix=PREFIX, tags=["Data"])
    app.include_router(models.router, prefix=PREFIX, tags=["Models"])
    app.include_router(simulation.router, prefix=PREFIX, tags=["Simulation"])
    app.include_router(recommendations.router, prefix=PREFIX, tags=["Recommendations"])
    app.include_router(copilot.router, prefix=PREFIX, tags=["Copilot"])

    return app


# Module-level app instance consumed by uvicorn and tests
app = create_app()
