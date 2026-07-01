"""Strategic recommendations router.

Real FastAPI endpoint that generates, prioritises and returns all 8 NovaMart
strategic recommendations ranked by RICE score.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Ensure src is importable when running via uvicorn from repo root
_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from api.schemas.recommendations import RecommendationListResponse, RecommendationResponse

router = APIRouter()


@router.get(
    "/recommendations",
    response_model=RecommendationListResponse,
    summary="Get prioritised strategic recommendations",
    description=(
        "Generate all NovaMart strategic recommendations from real data analysis, "
        "compute RICE scores, and return them sorted by business priority."
    ),
)
async def get_recommendations() -> RecommendationListResponse:
    """Return RICE-prioritised strategic recommendations for NovaMart leadership."""
    try:
        from bcgx.recommendations.engine import RecommendationEngine
        from bcgx.recommendations.prioritizer import RecommendationPrioritizer

        engine = RecommendationEngine()
        recs = engine.generate_all()
        prioritizer = RecommendationPrioritizer()
        ranked = prioritizer.prioritize(recs)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Data not available: {exc}. Run 'python scripts/generate_data.py' first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {exc}")

    rec_responses = [
        RecommendationResponse(
            id=r.id,
            title=r.title,
            category=r.category,
            description=r.description,
            priority=r.priority.value,
            expected_revenue_impact_usd=round(r.expected_revenue_impact_usd, 2),
            expected_profit_impact_usd=round(r.expected_profit_impact_usd, 2),
            roi=round(r.roi, 2),
            rice_score=round(r.rice_score, 2),
            timeline=r.timeline.value,
            confidence=r.confidence,
            difficulty=r.difficulty.value,
            implementation_effort_weeks=r.implementation_effort_weeks,
            risk=r.risk,
            evidence=r.evidence,
            reach=r.reach,
            kpis_to_track=r.kpis_to_track,
            dependencies=r.dependencies,
        )
        for r in ranked
    ]

    total_rev = sum(r.expected_revenue_impact_usd for r in ranked)
    total_profit = sum(r.expected_profit_impact_usd for r in ranked)

    return RecommendationListResponse(
        total=len(rec_responses),
        total_revenue_opportunity_usd=round(total_rev, 2),
        total_profit_opportunity_usd=round(total_profit, 2),
        recommendations=rec_responses,
    )
