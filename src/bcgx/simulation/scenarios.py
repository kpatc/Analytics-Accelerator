"""Pre-built scenario presets for the NovaMart simulation engine.

These presets represent the most business-relevant what-if analyses
surfaced by BCG X during the initial analytical engagement.
"""

from __future__ import annotations

from bcgx.simulation.engine import ScenarioInput, ScenarioType

PRESET_SCENARIOS: list[ScenarioInput] = [
    # 1. Small price increase on Electronics (elastic category — test volume sensitivity)
    ScenarioInput(
        scenario_type=ScenarioType.PRICE_CHANGE,
        parameters={"category": "Electronics", "price_change_pct": 0.05},
    ),
    # 2. Reallocate urban marketing toward digital (highest ROI channel in urban)
    ScenarioInput(
        scenario_type=ScenarioType.MARKETING_REALLOCATION,
        parameters={
            "reallocation": {
                "digital": 0.55,
                "tv": 0.15,
                "email": 0.20,
                "print": 0.05,
                "instore_promo": 0.05,
            },
            "store_format": "urban",
        },
    ),
    # 3. Silver tier loyalty rescue — 30% churn reduction program
    ScenarioInput(
        scenario_type=ScenarioType.CHURN_REDUCTION,
        parameters={
            "target_segment": "Silver",
            "churn_reduction_pct": 0.30,
            "intervention_cost_usd": 500_000,
        },
    ),
    # 4. Invest in 50 underperforming stores
    ScenarioInput(
        scenario_type=ScenarioType.STORE_INVESTMENT,
        parameters={
            "n_stores_to_invest": 50,
            "investment_per_store_usd": 150_000,
            "expected_performance_improvement_pct": 0.15,
        },
    ),
    # 5. Price increase on inelastic Food & Beverage category
    ScenarioInput(
        scenario_type=ScenarioType.PRICE_CHANGE,
        parameters={"category": "Food & Beverage", "price_change_pct": 0.05},
    ),
    # 6. Price increase on inelastic Health & Beauty category
    ScenarioInput(
        scenario_type=ScenarioType.PRICE_CHANGE,
        parameters={"category": "Health & Beauty", "price_change_pct": 0.05},
    ),
]
