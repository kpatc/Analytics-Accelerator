"""Pydantic schemas for the simulation API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PriceChangeRequest(BaseModel):
    """Request body for a price change simulation."""

    category: str = Field(..., description="Product category (e.g. 'Electronics')")
    price_change_pct: float = Field(
        ..., ge=-0.5, le=0.5, description="Fractional price change, e.g. 0.10 = +10%"
    )
    elasticity_override: float | None = Field(
        None, description="Override price elasticity coefficient (optional)"
    )


class MarketingReallocationRequest(BaseModel):
    """Request body for a marketing channel reallocation simulation."""

    reallocation: dict[str, float] = Field(
        ...,
        description=(
            "Channel -> fraction of budget allocation (values should sum to ~1.0). "
            "Valid channels: digital, tv, email, print, instore_promo"
        ),
    )
    total_budget_usd: float | None = Field(
        None, description="Total annual marketing budget in USD (defaults to current total)"
    )
    store_format: str = Field(
        "all",
        description="Store format filter: 'all' | 'urban' | 'rural' | 'suburban'",
    )


class ChurnReductionRequest(BaseModel):
    """Request body for a churn reduction simulation."""

    target_segment: str = Field(
        ...,
        description="Loyalty tier to target: 'Silver' | 'Bronze' | 'Gold' | 'All'",
    )
    churn_reduction_pct: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction by which churn is reduced, e.g. 0.30 = 30%"
    )
    intervention_cost_usd: float = Field(
        0.0, ge=0.0, description="One-time intervention cost in USD"
    )


class StoreInvestmentRequest(BaseModel):
    """Request body for a store investment simulation."""

    n_stores_to_invest: int = Field(..., ge=1, description="Number of stores to invest in")
    investment_per_store_usd: float = Field(
        ..., ge=0, description="Capital investment per store in USD"
    )
    expected_performance_improvement_pct: float = Field(
        ..., ge=0.0, le=1.0, description="Expected revenue improvement fraction, e.g. 0.15 = 15%"
    )


class ScenarioResponse(BaseModel):
    """Response body for any scenario simulation."""

    scenario_type: str
    projected_revenue: float
    projected_profit: float
    baseline_revenue: float
    baseline_profit: float
    revenue_delta: float
    profit_delta: float
    revenue_delta_pct: float
    profit_delta_pct: float
    projected_margin_pct: float
    baseline_margin_pct: float
    confidence_level: float
    assumptions: list[str]
    risks: list[str]
    timeline_months: int
