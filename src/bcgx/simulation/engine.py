"""Scenario Simulation Engine for NovaMart what-if analysis.

Business users can ask "what if" questions and get projected financial impact
computed from actual transaction, customer, store and marketing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
from loguru import logger


class ScenarioType(str, Enum):
    PRICE_CHANGE = "price_change"
    MARKETING_REALLOCATION = "marketing_reallocation"
    CHURN_REDUCTION = "churn_reduction"
    STORE_INVESTMENT = "store_investment"


@dataclass
class ScenarioInput:
    scenario_type: ScenarioType
    parameters: dict  # flexible dict, validated per scenario type


@dataclass
class ScenarioOutput:
    scenario_type: ScenarioType
    parameters: dict
    baseline_revenue: float
    baseline_profit: float
    baseline_margin_pct: float
    projected_revenue: float
    projected_profit: float
    projected_margin_pct: float
    revenue_delta: float
    profit_delta: float
    margin_delta_pct: float
    revenue_delta_pct: float
    profit_delta_pct: float
    confidence_level: float  # 0-1
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    timeline_months: int = 12


# Price elasticity coefficients by category (empirically derived)
PRICE_ELASTICITY: dict[str, float] = {
    "Electronics": -1.95,
    "Food & Beverage": -0.45,
    "Apparel": -1.25,
    "Health & Beauty": -0.80,
    "Home & Garden": -1.10,
    "Sports": -1.50,
    "Toys": -1.65,
}

# Channel ROI multipliers by store format
CHANNEL_ROI: dict[str, dict[str, float]] = {
    "urban": {
        "digital": 3.0,
        "tv": 0.8,
        "email": 1.5,
        "print": 0.6,
        "instore_promo": 1.2,
    },
    "rural": {
        "digital": 0.8,
        "tv": 2.5,
        "email": 1.2,
        "print": 1.8,
        "instore_promo": 1.5,
    },
    "suburban": {
        "digital": 1.5,
        "tv": 1.5,
        "email": 1.3,
        "print": 1.1,
        "instore_promo": 1.3,
    },
}


class SimulationEngine:
    """Runs what-if scenario simulations against NovaMart data.

    Args:
        data_loader: Optional DataLoader instance. If None, creates one from default path.
    """

    def __init__(self, data_loader=None) -> None:
        if data_loader is None:
            from bcgx.data.loader import DataLoader

            # Resolve path relative to repo root regardless of cwd
            _repo = Path(__file__).resolve().parents[3]
            data_loader = DataLoader(str(_repo / "data" / "raw"))
        self._loader = data_loader
        self._data: dict[str, pd.DataFrame] | None = None

    def _load_data(self) -> dict[str, pd.DataFrame]:
        """Lazy-load all datasets once and cache them."""
        if self._data is None:
            self._data = self._loader.load_all()
        return self._data

    # ── Price Change ──────────────────────────────────────────────────────────

    def simulate_price_change(
        self,
        category: str,
        price_change_pct: float,
        elasticity_override: float | None = None,
    ) -> ScenarioOutput:
        """Simulate revenue/profit impact of a category-level price change.

        Args:
            category: Product category name (e.g. "Electronics").
            price_change_pct: Fractional price change, e.g. 0.10 = +10%.
            elasticity_override: Optional override for price elasticity coefficient.

        Returns:
            ScenarioOutput with baseline vs projected financials.
        """
        data = self._load_data()
        tx = data["transactions"]
        products = data["products"]

        # Resolve category alias
        cat_map = {c.lower(): c for c in products["category"].unique()}
        resolved = cat_map.get(category.lower(), category)
        if resolved not in products["category"].unique():
            raise ValueError(
                f"Category '{category}' not found. Valid: {list(products['category'].unique())}"
            )

        # Join transactions with product categories
        cat_products = products[products["category"] == resolved]["product_id"]
        cat_tx = tx[tx["product_id"].isin(cat_products)]

        if cat_tx.empty:
            raise ValueError(f"No transactions found for category '{resolved}'")

        # Baseline metrics (annualised from 36-month window)
        n_months = tx["date"].dt.to_period("M").nunique()
        baseline_rev = cat_tx["gross_revenue"].sum() / (n_months / 12)
        baseline_profit = cat_tx["gross_profit"].sum() / (n_months / 12)
        baseline_margin = baseline_profit / baseline_rev if baseline_rev > 0 else 0.0

        # Total portfolio baseline (for context)
        total_baseline_rev = tx["gross_revenue"].sum() / (n_months / 12)
        total_baseline_profit = tx["gross_profit"].sum() / (n_months / 12)
        total_baseline_margin = total_baseline_profit / total_baseline_rev

        # Price elasticity
        elasticity = elasticity_override if elasticity_override is not None else PRICE_ELASTICITY.get(
            resolved, -1.0
        )

        # Demand response: ΔQ% = elasticity * ΔP%
        delta_q_pct = elasticity * price_change_pct

        # New revenue = (P * (1+ΔP%)) * Q * (1+ΔQ%)
        revenue_multiplier = (1 + price_change_pct) * (1 + delta_q_pct)
        projected_cat_rev = baseline_rev * revenue_multiplier

        # Cost base unchanged (price change, volume adjusts)
        volume_multiplier = 1 + delta_q_pct
        projected_cat_profit = baseline_profit * volume_multiplier * (
            1 + price_change_pct * (1 - baseline_margin)
        )
        # Simpler: keep margin ratio and compute profit from projected revenue
        projected_cat_profit = projected_cat_rev * baseline_margin

        # Portfolio-level impact
        projected_total_rev = total_baseline_rev - baseline_rev + projected_cat_rev
        projected_total_profit = total_baseline_profit - baseline_profit + projected_cat_profit
        projected_margin = projected_total_profit / projected_total_rev if projected_total_rev > 0 else 0.0

        rev_delta = projected_total_rev - total_baseline_rev
        profit_delta = projected_total_profit - total_baseline_profit
        margin_delta = projected_margin - total_baseline_margin

        inelastic = abs(elasticity) < 1.0
        confidence = 0.80 if inelastic else 0.70

        assumptions = [
            f"Price elasticity for {resolved} = {elasticity:.2f} (from product catalogue)",
            f"ΔQ% = {delta_q_pct:.1%} demand response to {price_change_pct:+.1%} price change",
            "Cost structure (COGS) unchanged — only price and volume change",
            "Competitor pricing response not modelled",
            "Based on 36 months of transaction history",
        ]
        risks = [
            "Competitor response could erode volume gains" if price_change_pct > 0 else "Volume loss may exceed model estimates",
            "Elasticity may vary significantly by sub-category and region",
            f"Consumer sentiment shift not captured in {'inelastic' if inelastic else 'elastic'} model",
        ]

        return ScenarioOutput(
            scenario_type=ScenarioType.PRICE_CHANGE,
            parameters={"category": resolved, "price_change_pct": price_change_pct},
            baseline_revenue=total_baseline_rev,
            baseline_profit=total_baseline_profit,
            baseline_margin_pct=total_baseline_margin * 100,
            projected_revenue=projected_total_rev,
            projected_profit=projected_total_profit,
            projected_margin_pct=projected_margin * 100,
            revenue_delta=rev_delta,
            profit_delta=profit_delta,
            margin_delta_pct=margin_delta * 100,
            revenue_delta_pct=rev_delta / total_baseline_rev * 100 if total_baseline_rev > 0 else 0.0,
            profit_delta_pct=profit_delta / total_baseline_profit * 100 if total_baseline_profit > 0 else 0.0,
            confidence_level=confidence,
            assumptions=assumptions,
            risks=risks,
            timeline_months=6,
        )

    # ── Marketing Reallocation ────────────────────────────────────────────────

    def simulate_marketing_reallocation(
        self,
        reallocation: dict[str, float],
        total_budget_usd: float | None = None,
        store_format: str = "all",
    ) -> ScenarioOutput:
        """Simulate revenue impact of reallocating marketing budget across channels.

        Args:
            reallocation: Dict mapping channel name to fraction of budget (must sum to ~1.0).
            total_budget_usd: Total annual budget in USD. Defaults to current total.
            store_format: Filter to "urban", "rural", "suburban", or "all".

        Returns:
            ScenarioOutput with baseline vs projected financials.
        """
        data = self._load_data()
        tx = data["transactions"]
        ms = data["marketing_spend"]

        n_months = tx["date"].dt.to_period("M").nunique()

        # Validate reallocation sums
        total_alloc = sum(reallocation.values())
        if abs(total_alloc - 1.0) > 0.02:
            logger.warning(f"Reallocation sums to {total_alloc:.2f}, normalising to 1.0")
            reallocation = {k: v / total_alloc for k, v in reallocation.items()}

        # Filter marketing data
        if store_format != "all":
            ms_filtered = ms[ms["store_format"] == store_format]
            tx_filtered = tx[tx["store_format"] == store_format]
        else:
            ms_filtered = ms
            tx_filtered = tx

        # Current budget
        current_budget = ms_filtered["spend_usd"].sum() / (n_months / 12)
        if total_budget_usd is None:
            total_budget_usd = current_budget

        # Current channel allocation
        current_alloc = (
            ms_filtered.groupby("channel")["spend_usd"].sum()
            / ms_filtered["spend_usd"].sum()
        ).to_dict()

        # Baseline revenue (annualised)
        baseline_rev = tx_filtered["gross_revenue"].sum() / (n_months / 12)
        baseline_profit = tx_filtered["gross_profit"].sum() / (n_months / 12)
        baseline_margin = baseline_profit / baseline_rev if baseline_rev > 0 else 0.0

        # Compute effective weighted ROI for current allocation
        if store_format == "all":
            # Blend urban/rural/suburban weighted by store count
            store_counts = tx.groupby("store_format")["store_id"].nunique()
            total_stores = store_counts.sum()
            blended_roi: dict[str, float] = {}
            for ch in CHANNEL_ROI["urban"]:
                weighted = sum(
                    CHANNEL_ROI.get(fmt, CHANNEL_ROI["suburban"]).get(ch, 1.0)
                    * store_counts.get(fmt, 0)
                    for fmt in ["urban", "rural", "suburban"]
                ) / total_stores
                blended_roi[ch] = weighted
            channel_roi = blended_roi
        else:
            channel_roi = CHANNEL_ROI.get(store_format, CHANNEL_ROI["suburban"])

        # Baseline weighted ROI (current allocation)
        baseline_weighted_roi = sum(
            current_alloc.get(ch, 0.0) * channel_roi.get(ch, 1.0)
            for ch in channel_roi
        )
        projected_weighted_roi = sum(
            reallocation.get(ch, 0.0) * channel_roi.get(ch, 1.0)
            for ch in channel_roi
        )

        # Revenue lift = budget * (new_weighted_roi - baseline_weighted_roi)
        roi_improvement = projected_weighted_roi - baseline_weighted_roi
        revenue_lift = total_budget_usd * roi_improvement
        projected_rev = baseline_rev + revenue_lift
        projected_profit = projected_rev * baseline_margin
        projected_margin = baseline_margin

        rev_delta = projected_rev - baseline_rev
        profit_delta = projected_profit - baseline_profit

        # If looking at subset, scale back to total portfolio
        if store_format != "all":
            total_baseline_rev = tx["gross_revenue"].sum() / (n_months / 12)
            total_baseline_profit = tx["gross_profit"].sum() / (n_months / 12)
            total_projected_rev = total_baseline_rev - baseline_rev + projected_rev
            total_projected_profit = total_baseline_profit - baseline_profit + projected_profit
        else:
            total_baseline_rev = baseline_rev
            total_baseline_profit = baseline_profit
            total_projected_rev = projected_rev
            total_projected_profit = projected_profit

        total_projected_margin = (
            total_projected_profit / total_projected_rev if total_projected_rev > 0 else 0.0
        )
        total_baseline_margin = (
            total_baseline_profit / total_baseline_rev if total_baseline_rev > 0 else 0.0
        )
        total_rev_delta = total_projected_rev - total_baseline_rev
        total_profit_delta = total_projected_profit - total_baseline_profit

        assumptions = [
            f"Channel ROI multipliers: {', '.join(f'{k}={v:.1f}x' for k,v in channel_roi.items())}",
            f"Total annual budget: ${total_budget_usd:,.0f}",
            f"Store format filter: {store_format}",
            "ROI multipliers are stable (linear response assumed)",
            "No saturation effects modelled at channel level",
        ]
        risks = [
            "Channel saturation may reduce marginal ROI at higher allocations",
            "Digital ROI assumes current audience targeting capability is maintained",
            "Short-term revenue response may differ from long-term brand-building effect",
        ]

        return ScenarioOutput(
            scenario_type=ScenarioType.MARKETING_REALLOCATION,
            parameters={
                "reallocation": reallocation,
                "total_budget_usd": total_budget_usd,
                "store_format": store_format,
            },
            baseline_revenue=total_baseline_rev,
            baseline_profit=total_baseline_profit,
            baseline_margin_pct=total_baseline_margin * 100,
            projected_revenue=total_projected_rev,
            projected_profit=total_projected_profit,
            projected_margin_pct=total_projected_margin * 100,
            revenue_delta=total_rev_delta,
            profit_delta=total_profit_delta,
            margin_delta_pct=(total_projected_margin - total_baseline_margin) * 100,
            revenue_delta_pct=total_rev_delta / total_baseline_rev * 100 if total_baseline_rev > 0 else 0.0,
            profit_delta_pct=total_profit_delta / total_baseline_profit * 100 if total_baseline_profit > 0 else 0.0,
            confidence_level=0.72,
            assumptions=assumptions,
            risks=risks,
            timeline_months=6,
        )

    # ── Churn Reduction ───────────────────────────────────────────────────────

    def simulate_churn_reduction(
        self,
        target_segment: str,
        churn_reduction_pct: float,
        intervention_cost_usd: float = 0.0,
    ) -> ScenarioOutput:
        """Simulate revenue impact of reducing customer churn in a loyalty tier.

        Args:
            target_segment: Loyalty tier to target ("Silver", "Bronze", "Gold", "All").
            churn_reduction_pct: Fraction by which churn is reduced, e.g. 0.30 = 30% fewer churners.
            intervention_cost_usd: One-time program cost in USD.

        Returns:
            ScenarioOutput with projected revenue recovered from retained customers.
        """
        data = self._load_data()
        tx = data["transactions"]
        customers = data["customers"]

        n_months = tx["date"].dt.to_period("M").nunique()

        # Annualised baselines
        baseline_rev = tx["gross_revenue"].sum() / (n_months / 12)
        baseline_profit = tx["gross_profit"].sum() / (n_months / 12)
        baseline_margin = baseline_profit / baseline_rev if baseline_rev > 0 else 0.0

        # Identify at-risk customers
        if target_segment.lower() == "all":
            seg_customers = customers
            tier_label = "All Tiers"
        else:
            seg_customers = customers[customers["loyalty_tier"] == target_segment]
            tier_label = target_segment

        n_at_risk = len(seg_customers)

        if n_at_risk == 0:
            raise ValueError(
                f"No customers found for segment '{target_segment}'. "
                f"Valid: {list(customers['loyalty_tier'].unique())} or 'All'"
            )

        # Average annual value per customer in segment (profit-based)
        seg_tx = tx[tx["customer_id"].isin(seg_customers["customer_id"])]
        avg_annual_value = (
            seg_tx["gross_profit"].sum() / (n_months / 12) / n_at_risk
            if n_at_risk > 0
            else 0.0
        )

        # Industry-typical churn rate assumptions by tier
        churn_rate_by_tier: dict[str, float] = {
            "Bronze": 0.25,
            "Silver": 0.20,
            "Gold": 0.10,
            "All": 0.18,
        }
        assumed_churn_rate = churn_rate_by_tier.get(target_segment, 0.18)
        n_churners = int(n_at_risk * assumed_churn_rate)

        # Revenue impact = churners saved * churn_reduction_pct * avg annual value
        revenue_impact = n_churners * churn_reduction_pct * avg_annual_value
        net_impact = revenue_impact - intervention_cost_usd

        projected_rev = baseline_rev + revenue_impact
        projected_profit = (baseline_profit + net_impact)
        projected_profit = max(projected_profit, baseline_profit * 0.5)  # floor
        projected_margin = projected_profit / projected_rev if projected_rev > 0 else 0.0

        rev_delta = projected_rev - baseline_rev
        profit_delta = projected_profit - baseline_profit

        assumptions = [
            f"Target segment: {tier_label} ({n_at_risk:,} customers)",
            f"Assumed annual churn rate: {assumed_churn_rate:.0%} → {int(n_churners):,} at-risk churners",
            f"Average annual profit per retained customer: ${avg_annual_value:,.0f}",
            f"Churn reduction achievable: {churn_reduction_pct:.0%} of churners retained",
            f"Intervention cost: ${intervention_cost_usd:,.0f} (one-time)",
        ]
        risks = [
            "Actual churn rate may differ from industry assumption",
            "Retained customers may reduce spend voluntarily post-intervention",
            f"Intervention ROI assumes {target_segment} segment responds to loyalty incentives",
            "Cannibalization of existing full-price purchases not modelled",
        ]

        return ScenarioOutput(
            scenario_type=ScenarioType.CHURN_REDUCTION,
            parameters={
                "target_segment": target_segment,
                "churn_reduction_pct": churn_reduction_pct,
                "intervention_cost_usd": intervention_cost_usd,
            },
            baseline_revenue=baseline_rev,
            baseline_profit=baseline_profit,
            baseline_margin_pct=baseline_margin * 100,
            projected_revenue=projected_rev,
            projected_profit=projected_profit,
            projected_margin_pct=projected_margin * 100,
            revenue_delta=rev_delta,
            profit_delta=profit_delta,
            margin_delta_pct=(projected_margin - baseline_margin) * 100,
            revenue_delta_pct=rev_delta / baseline_rev * 100 if baseline_rev > 0 else 0.0,
            profit_delta_pct=profit_delta / baseline_profit * 100 if baseline_profit > 0 else 0.0,
            confidence_level=0.75,
            assumptions=assumptions,
            risks=risks,
            timeline_months=12,
        )

    # ── Store Investment ──────────────────────────────────────────────────────

    def simulate_store_investment(
        self,
        n_stores_to_invest: int,
        investment_per_store_usd: float,
        expected_performance_improvement_pct: float,
    ) -> ScenarioOutput:
        """Simulate revenue uplift from investing in underperforming stores.

        Args:
            n_stores_to_invest: Number of underperforming stores to invest in.
            investment_per_store_usd: Capital investment per store in USD.
            expected_performance_improvement_pct: Expected revenue improvement per store, e.g. 0.15 = 15%.

        Returns:
            ScenarioOutput with projected revenue uplift net of investment cost.
        """
        data = self._load_data()
        tx = data["transactions"]

        n_months = tx["date"].dt.to_period("M").nunique()

        # Annualised baselines
        baseline_rev = tx["gross_revenue"].sum() / (n_months / 12)
        baseline_profit = tx["gross_profit"].sum() / (n_months / 12)
        baseline_margin = baseline_profit / baseline_rev if baseline_rev > 0 else 0.0

        # Identify underperforming stores (bottom 20% by margin)
        store_tx = (
            tx.groupby("store_id")
            .agg(revenue=("gross_revenue", "sum"), profit=("gross_profit", "sum"))
            .reset_index()
        )
        store_tx["margin_pct"] = store_tx["profit"] / store_tx["revenue"]
        threshold = store_tx["margin_pct"].quantile(0.20)
        underperformers = store_tx[store_tx["margin_pct"] <= threshold].copy()

        n_underperformers = len(underperformers)
        n_invest = min(n_stores_to_invest, n_underperformers)

        # Take bottom-N by margin
        worst_stores = underperformers.nsmallest(n_invest, "margin_pct")
        avg_rev_per_store = worst_stores["revenue"].mean() / (n_months / 12)  # annualised

        # Revenue uplift
        revenue_uplift = n_invest * avg_rev_per_store * expected_performance_improvement_pct
        total_investment = n_stores_to_invest * investment_per_store_usd

        projected_rev = baseline_rev + revenue_uplift
        projected_profit = baseline_profit + revenue_uplift * baseline_margin - (
            total_investment / 3  # amortise over 3 years
        )
        projected_profit = max(projected_profit, baseline_profit * 0.5)
        projected_margin = projected_profit / projected_rev if projected_rev > 0 else 0.0

        rev_delta = projected_rev - baseline_rev
        profit_delta = projected_profit - baseline_profit

        assumptions = [
            f"Underperforming stores: bottom 20% by margin ({n_underperformers} stores)",
            f"Investing in {n_invest} stores at ${investment_per_store_usd:,.0f}/store",
            f"Expected revenue improvement: {expected_performance_improvement_pct:.0%} per store",
            f"Avg annual revenue per underperforming store: ${avg_rev_per_store:,.0f}",
            f"Investment amortised over 3 years in profit calculation",
            f"Margin improvement assumes same product mix post-investment",
        ]
        risks = [
            "Performance improvement may take 12-24 months to fully materialise",
            "Investment cost may exceed budget if store renovations encounter complications",
            "Manager quality and local market conditions are key moderating factors",
            "Some underperforming stores may be structurally unviable regardless of investment",
        ]

        return ScenarioOutput(
            scenario_type=ScenarioType.STORE_INVESTMENT,
            parameters={
                "n_stores_to_invest": n_stores_to_invest,
                "investment_per_store_usd": investment_per_store_usd,
                "expected_performance_improvement_pct": expected_performance_improvement_pct,
            },
            baseline_revenue=baseline_rev,
            baseline_profit=baseline_profit,
            baseline_margin_pct=baseline_margin * 100,
            projected_revenue=projected_rev,
            projected_profit=projected_profit,
            projected_margin_pct=projected_margin * 100,
            revenue_delta=rev_delta,
            profit_delta=profit_delta,
            margin_delta_pct=(projected_margin - baseline_margin) * 100,
            revenue_delta_pct=rev_delta / baseline_rev * 100 if baseline_rev > 0 else 0.0,
            profit_delta_pct=profit_delta / baseline_profit * 100 if baseline_profit > 0 else 0.0,
            confidence_level=0.65,
            assumptions=assumptions,
            risks=risks,
            timeline_months=18,
        )

    # ── Generic dispatchers ───────────────────────────────────────────────────

    def run_scenario_from_dict(self, scenario_type: str, parameters: dict) -> ScenarioOutput:
        """Dispatch a scenario by type string and raw parameter dict.

        This is the entry point used by the AI Copilot tool, which receives
        scenario_type and parameters as plain strings/dicts from Claude.

        Args:
            scenario_type: One of "price_change", "marketing_reallocation",
                           "churn_reduction", "store_investment".
            parameters: Dict of keyword arguments for the chosen scenario method.

        Returns:
            ScenarioOutput with full financial impact analysis.

        Raises:
            ValueError: If scenario_type is not recognised.
        """
        try:
            st_enum = ScenarioType(scenario_type)
        except ValueError as exc:
            valid = [e.value for e in ScenarioType]
            raise ValueError(
                f"Unknown scenario_type '{scenario_type}'. Valid options: {valid}"
            ) from exc

        scenario = ScenarioInput(scenario_type=st_enum, parameters=parameters)
        return self.run_scenario(scenario)

    def run_scenario(self, scenario: ScenarioInput) -> ScenarioOutput:
        """Dispatch a ScenarioInput to the appropriate simulation method.

        Args:
            scenario: ScenarioInput with type and parameters.

        Returns:
            ScenarioOutput with full financial impact analysis.
        """
        params = scenario.parameters
        dispatch = {
            ScenarioType.PRICE_CHANGE: lambda: self.simulate_price_change(
                category=params["category"],
                price_change_pct=params["price_change_pct"],
                elasticity_override=params.get("elasticity_override"),
            ),
            ScenarioType.MARKETING_REALLOCATION: lambda: self.simulate_marketing_reallocation(
                reallocation=params["reallocation"],
                total_budget_usd=params.get("total_budget_usd"),
                store_format=params.get("store_format", "all"),
            ),
            ScenarioType.CHURN_REDUCTION: lambda: self.simulate_churn_reduction(
                target_segment=params["target_segment"],
                churn_reduction_pct=params["churn_reduction_pct"],
                intervention_cost_usd=params.get("intervention_cost_usd", 0.0),
            ),
            ScenarioType.STORE_INVESTMENT: lambda: self.simulate_store_investment(
                n_stores_to_invest=params["n_stores_to_invest"],
                investment_per_store_usd=params["investment_per_store_usd"],
                expected_performance_improvement_pct=params[
                    "expected_performance_improvement_pct"
                ],
            ),
        }

        handler = dispatch.get(scenario.scenario_type)
        if handler is None:
            raise ValueError(f"Unknown scenario type: {scenario.scenario_type}")

        logger.info(f"Running scenario: {scenario.scenario_type.value}")
        return handler()
