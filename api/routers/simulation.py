"""Scenario simulation router.

Real FastAPI endpoints that execute what-if business simulations using the
SimulationEngine and return projected financial impact.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Ensure src is importable when running via uvicorn from repo root
_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from api.schemas.simulation import (
    ChurnReductionRequest,
    MarketingReallocationRequest,
    PriceChangeRequest,
    ScenarioResponse,
    StoreInvestmentRequest,
)

router = APIRouter()


def _to_response(result) -> ScenarioResponse:
    """Convert a ScenarioOutput to the API response schema."""
    return ScenarioResponse(
        scenario_type=result.scenario_type.value,
        projected_revenue=round(result.projected_revenue, 2),
        projected_profit=round(result.projected_profit, 2),
        baseline_revenue=round(result.baseline_revenue, 2),
        baseline_profit=round(result.baseline_profit, 2),
        revenue_delta=round(result.revenue_delta, 2),
        profit_delta=round(result.profit_delta, 2),
        revenue_delta_pct=round(result.revenue_delta_pct, 4),
        profit_delta_pct=round(result.profit_delta_pct, 4),
        projected_margin_pct=round(result.projected_margin_pct, 4),
        baseline_margin_pct=round(result.baseline_margin_pct, 4),
        confidence_level=result.confidence_level,
        assumptions=result.assumptions,
        risks=result.risks,
        timeline_months=result.timeline_months,
    )


def _get_engine():
    """Create a SimulationEngine, raising 503 if data is unavailable."""
    try:
        from bcgx.simulation.engine import SimulationEngine

        return SimulationEngine()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Simulation engine unavailable: {exc}. Run 'python scripts/generate_data.py' first.",
        )


@router.post(
    "/simulation/price-change",
    response_model=ScenarioResponse,
    summary="Simulate a category price change",
    description=(
        "Run a what-if simulation for a price change on a specific product category. "
        "Returns projected revenue, profit and margin impact using price elasticity."
    ),
)
async def simulate_price_change(request: PriceChangeRequest) -> ScenarioResponse:
    """Simulate revenue/profit impact of a category price change."""
    engine = _get_engine()
    try:
        result = engine.simulate_price_change(
            category=request.category,
            price_change_pct=request.price_change_pct,
            elasticity_override=request.elasticity_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")
    return _to_response(result)


@router.post(
    "/simulation/marketing",
    response_model=ScenarioResponse,
    summary="Simulate a marketing channel reallocation",
    description=(
        "Run a what-if simulation for reallocating the marketing budget across channels. "
        "Returns projected revenue impact based on channel ROI multipliers."
    ),
)
async def simulate_marketing(request: MarketingReallocationRequest) -> ScenarioResponse:
    """Simulate revenue impact of reallocating marketing budget."""
    engine = _get_engine()
    try:
        result = engine.simulate_marketing_reallocation(
            reallocation=request.reallocation,
            total_budget_usd=request.total_budget_usd,
            store_format=request.store_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")
    return _to_response(result)


@router.post(
    "/simulation/churn",
    response_model=ScenarioResponse,
    summary="Simulate a churn reduction programme",
    description=(
        "Run a what-if simulation for reducing customer churn in a loyalty tier. "
        "Returns projected revenue and profit from retained customers."
    ),
)
async def simulate_churn(request: ChurnReductionRequest) -> ScenarioResponse:
    """Simulate revenue/profit impact of a churn reduction programme."""
    engine = _get_engine()
    try:
        result = engine.simulate_churn_reduction(
            target_segment=request.target_segment,
            churn_reduction_pct=request.churn_reduction_pct,
            intervention_cost_usd=request.intervention_cost_usd,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")
    return _to_response(result)


@router.post(
    "/simulation/store-investment",
    response_model=ScenarioResponse,
    summary="Simulate a store investment programme",
    description=(
        "Run a what-if simulation for investing capital in underperforming stores. "
        "Returns projected revenue uplift net of investment cost."
    ),
)
async def simulate_store_investment(request: StoreInvestmentRequest) -> ScenarioResponse:
    """Simulate revenue uplift from investing in underperforming stores."""
    engine = _get_engine()
    try:
        result = engine.simulate_store_investment(
            n_stores_to_invest=request.n_stores_to_invest,
            investment_per_store_usd=request.investment_per_store_usd,
            expected_performance_improvement_pct=request.expected_performance_improvement_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")
    return _to_response(result)


# Legacy endpoint kept for backwards compatibility
@router.post(
    "/simulation/run",
    response_model=ScenarioResponse,
    summary="Run a generic scenario simulation",
    deprecated=True,
)
async def run_simulation_legacy(request: PriceChangeRequest) -> ScenarioResponse:
    """Legacy simulation endpoint — use /simulation/price-change instead."""
    return await simulate_price_change(request)
