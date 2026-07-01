"""Pydantic schemas for the recommendations API endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    """Response schema for a single strategic recommendation."""

    id: str
    title: str
    category: str
    description: str
    priority: str
    expected_revenue_impact_usd: float
    expected_profit_impact_usd: float
    roi: float
    rice_score: float
    timeline: str
    confidence: float
    difficulty: str
    implementation_effort_weeks: int
    risk: str
    evidence: str
    reach: int
    kpis_to_track: list[str]
    dependencies: list[str]


class RecommendationListResponse(BaseModel):
    """Response schema for the full recommendations list endpoint."""

    total: int
    total_revenue_opportunity_usd: float
    total_profit_opportunity_usd: float
    recommendations: list[RecommendationResponse]
