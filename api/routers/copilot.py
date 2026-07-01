"""AI Executive Copilot API router.

Exposes the Claude-powered analytics copilot as REST endpoints consumed by
the Streamlit dashboard and any external client.
"""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class CopilotQuery(BaseModel):
    """Request body for a copilot query."""

    question: str = Field(..., description="Natural-language business question about NovaMart")
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Prior messages for multi-turn conversation ({role, content} dicts)",
    )


class CopilotAnswer(BaseModel):
    """Response from the copilot."""

    answer: str = Field(..., description="Markdown-formatted answer grounded in NovaMart data")
    tools_called: list[str] = Field(default_factory=list, description="Analytics tools invoked")
    sources: list[str] = Field(default_factory=list, description="Data sources cited")
    is_configured: bool = Field(..., description="True if the Anthropic API key is set")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/copilot/query",
    response_model=CopilotAnswer,
    summary="Query the Executive Analytics Copilot",
    description=(
        "Submit a natural-language question and receive a data-grounded answer from "
        "the Claude-powered copilot.  The copilot calls analytics tools automatically "
        "to retrieve real NovaMart metrics before generating its response."
    ),
)
async def query_copilot(request: CopilotQuery) -> CopilotAnswer:
    """Handle a copilot query end-to-end."""
    from bcgx.copilot.agent import ExecutiveCopilot

    copilot = ExecutiveCopilot()

    if not copilot.is_configured():
        logger.info("Copilot query received but ANTHROPIC_API_KEY not configured")
        return CopilotAnswer(
            answer=(
                "**AI Copilot Not Configured**\n\n"
                "The Executive Analytics Copilot requires an Anthropic API key.\n\n"
                "Set `ANTHROPIC_API_KEY` in your `.env` file and restart the server."
            ),
            tools_called=[],
            sources=[],
            is_configured=False,
        )

    logger.info(f"Copilot query: {request.question[:80]}...")
    response = copilot.ask(request.question, request.conversation_history)

    return CopilotAnswer(
        answer=response.answer,
        tools_called=response.tools_called,
        sources=response.sources,
        is_configured=True,
    )


@router.get(
    "/copilot/examples",
    response_model=list[str],
    summary="Get example copilot questions",
    description="Returns a curated list of business questions the copilot can answer.",
)
async def get_example_questions() -> list[str]:
    """Return example questions for the copilot UI."""
    from bcgx.copilot.prompts import EXAMPLE_QUESTIONS

    return EXAMPLE_QUESTIONS


@router.get(
    "/copilot/status",
    summary="Check copilot configuration status",
)
async def copilot_status() -> dict:
    """Return whether the copilot is fully configured and ready."""
    from bcgx.copilot.agent import ExecutiveCopilot
    from bcgx.config.settings import get_settings

    settings = get_settings()
    copilot = ExecutiveCopilot()

    return {
        "configured": copilot.is_configured(),
        "model": settings.anthropic.model,
        "tools_available": [
            "get_financial_summary",
            "get_churn_analysis",
            "get_store_performance",
            "get_marketing_roi",
            "get_pricing_analysis",
            "get_recommendations",
            "simulate_scenario",
        ],
    }
