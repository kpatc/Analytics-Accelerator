"""Recommendation data schema for NovaMart strategic recommendations.

Defines the Recommendation dataclass and supporting enumerations used by
the recommendation engine, prioritiser and downstream consumers (API, slides).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Priority(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class TimelineCategory(str, Enum):
    IMMEDIATE = "0-30 days"
    SHORT_TERM = "1-3 months"
    MEDIUM_TERM = "3-6 months"
    LONG_TERM = "6-12 months"


@dataclass
class Recommendation:
    """A single strategic recommendation with evidence, impact and prioritisation metadata.

    Attributes:
        id: Unique identifier (e.g. "REC-001").
        title: Short descriptive title (< 80 chars).
        category: Analytical domain — "Pricing" | "Marketing" | "Retention" |
                  "Operations" | "Store Portfolio".
        description: 2-3 sentence business-facing description.
        evidence: Evidence statement, e.g. "Based on statistical analysis showing..."
        expected_revenue_impact_usd: Annual revenue uplift in USD.
        expected_profit_impact_usd: Annual profit uplift in USD.
        confidence: Model confidence in the estimate (0-1).
        difficulty: Implementation complexity assessment.
        implementation_effort_weeks: Estimated person-weeks of effort.
        timeline: When the initiative can realistically be delivered.
        risk: Primary risk statement.
        priority: Business priority classification.
        roi: Return on investment multiple (profit_impact / implementation_cost).
        reach: Number of customers/stores directly affected.
        impact_score: Subjective impact score 1-10 for RICE scoring.
        rice_score: Reach × Impact × Confidence / Effort (higher = more attractive).
        dependencies: List of recommendation IDs this depends on.
        kpis_to_track: List of KPI names to monitor post-implementation.
    """

    id: str
    title: str
    category: str  # "Pricing" | "Marketing" | "Retention" | "Operations" | "Store Portfolio"
    description: str
    evidence: str
    expected_revenue_impact_usd: float
    expected_profit_impact_usd: float
    confidence: float  # 0-1
    difficulty: Difficulty
    implementation_effort_weeks: int
    timeline: TimelineCategory
    risk: str
    priority: Priority
    roi: float  # profit_impact / implementation_cost
    reach: int  # customers or stores affected
    impact_score: float  # 1-10
    rice_score: float  # Reach * Impact * Confidence / Effort
    dependencies: list[str] = field(default_factory=list)
    kpis_to_track: list[str] = field(default_factory=list)
