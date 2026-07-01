"""Analytics tools for the Executive Copilot.

Each function retrieves real metrics from NovaMart's data and returns
a JSON-serialisable dict.  Claude calls these tools automatically when
answering business questions so every answer is grounded in actual numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from anthropic.types import ToolParam
from loguru import logger


# ── Shared data-loading helper ────────────────────────────────────────────────

def _load_data() -> dict[str, pd.DataFrame]:
    """Load all NovaMart datasets.  Lazy helper used by every tool function."""
    from bcgx.data.loader import DataLoader

    _repo = Path(__file__).resolve().parents[3]
    loader = DataLoader(str(_repo / "data" / "raw"))
    return loader.load_all()


# ── Tool functions ─────────────────────────────────────────────────────────────


def get_financial_summary() -> dict[str, Any]:
    """Get NovaMart's key financial KPIs.

    Returns:
        Dict with total_revenue_usd, total_profit_usd, avg_margin_pct,
        margin_start_pct (months 1-3), margin_end_pct (months 34-36),
        margin_change_pp, total_stores, total_customers.
    """
    try:
        data = _load_data()
        tx = data["transactions"]

        # Month index (1-based)
        min_period = tx["date"].dt.to_period("M").min()
        tx = tx.copy()
        tx["month_num"] = (
            tx["date"].dt.to_period("M").apply(lambda p: (p - min_period).n + 1)
        )

        total_rev = float(tx["gross_revenue"].sum())
        total_profit = float(tx["gross_profit"].sum())
        avg_margin = total_profit / total_rev if total_rev > 0 else 0.0

        early = tx[tx["month_num"] <= 3]
        late = tx[tx["month_num"] >= 34]
        margin_start = (
            float(early["gross_profit"].sum() / early["gross_revenue"].sum())
            if not early.empty else avg_margin
        )
        margin_end = (
            float(late["gross_profit"].sum() / late["gross_revenue"].sum())
            if not late.empty else avg_margin
        )

        n_months = tx["date"].dt.to_period("M").nunique()
        annualised_rev = total_rev / (n_months / 12)
        annualised_profit = total_profit / (n_months / 12)

        return {
            "total_revenue_usd": round(total_rev, 2),
            "total_profit_usd": round(total_profit, 2),
            "annualised_revenue_usd": round(annualised_rev, 2),
            "annualised_profit_usd": round(annualised_profit, 2),
            "avg_margin_pct": round(avg_margin * 100, 2),
            "margin_start_pct": round(margin_start * 100, 2),
            "margin_end_pct": round(margin_end * 100, 2),
            "margin_change_pp": round((margin_end - margin_start) * 100, 2),
            "total_stores": int(data["stores"].shape[0]),
            "total_customers": int(data["customers"].shape[0]),
            "total_transactions": int(tx.shape[0]),
            "data_period_months": int(n_months),
        }
    except Exception as exc:
        logger.error(f"get_financial_summary failed: {exc}")
        return {"error": str(exc)}


def get_churn_analysis() -> dict[str, Any]:
    """Get customer churn analysis by loyalty tier.

    Returns:
        Dict with overall_churn_rate, churn_by_tier, silver_churn_pre/post month 24,
        revenue_at_risk_usd, most_churned_segment.
    """
    try:
        data = _load_data()
        tx = data["transactions"]
        customers = data["customers"]

        tx = tx.copy()
        tx["date_pd"] = pd.to_datetime(tx["date"])

        # Approximated churn: no purchase in final 6 months of the dataset
        max_date = tx["date_pd"].max()
        cutoff = max_date - pd.Timedelta(days=180)
        last_purchase = tx.groupby("customer_id")["date_pd"].max()
        churned_ids = set(last_purchase[last_purchase < cutoff].index)

        cust_tier = customers.set_index("customer_id")["loyalty_tier"]

        churn_by_tier: dict[str, float] = {}
        churn_counts: dict[str, int] = {}
        for tier in ["Gold", "Silver", "Bronze"]:
            tier_ids = set(cust_tier[cust_tier == tier].index)
            n_tier = len(tier_ids)
            n_churned = len(churned_ids & tier_ids)
            churn_by_tier[tier] = round(n_churned / n_tier, 4) if n_tier > 0 else 0.0
            churn_counts[tier] = n_churned

        overall_rate = len(churned_ids) / len(customers)

        # Silver churn pre/post month 24
        min_period = tx["date_pd"].dt.to_period("M").min()
        tx["month_num"] = tx["date_pd"].dt.to_period("M").apply(
            lambda p: (p - min_period).n + 1
        )
        silver_ids = set(cust_tier[cust_tier == "Silver"].index)

        tx_pre = tx[tx["month_num"] <= 24]
        tx_post = tx[tx["month_num"] > 24]

        silver_active_pre = set(tx_pre[tx_pre["customer_id"].isin(silver_ids)]["customer_id"])
        silver_active_post = set(tx_post[tx_post["customer_id"].isin(silver_ids)]["customer_id"])
        silver_churn_pre = 1 - len(silver_active_pre) / len(silver_ids)
        silver_churn_post = 1 - len(silver_active_post) / len(silver_ids)

        # Revenue at risk from Silver churners
        silver_tx = tx[tx["customer_id"].isin(silver_ids)]
        n_months = tx["date_pd"].dt.to_period("M").nunique()
        avg_annual_rev_silver = (
            silver_tx["gross_revenue"].sum() / (n_months / 12) / len(silver_ids)
            if len(silver_ids) > 0 else 0.0
        )
        n_silver_churners = churn_counts.get("Silver", 0)
        revenue_at_risk = avg_annual_rev_silver * n_silver_churners

        most_churned = max(churn_by_tier, key=lambda k: churn_by_tier[k])

        return {
            "overall_churn_rate": round(overall_rate, 4),
            "churn_by_tier": {k: round(v, 4) for k, v in churn_by_tier.items()},
            "churn_count_by_tier": churn_counts,
            "silver_churn_pre_month24": round(silver_churn_pre, 4),
            "silver_churn_post_month24": round(silver_churn_post, 4),
            "silver_churn_acceleration_pp": round((silver_churn_post - silver_churn_pre) * 100, 2),
            "revenue_at_risk_usd": round(revenue_at_risk, 2),
            "avg_annual_revenue_per_silver_customer_usd": round(avg_annual_rev_silver, 2),
            "most_churned_segment": most_churned,
            "total_silver_customers": int(len(silver_ids)),
            "total_silver_churners": int(n_silver_churners),
        }
    except Exception as exc:
        logger.error(f"get_churn_analysis failed: {exc}")
        return {"error": str(exc)}


def get_store_performance() -> dict[str, Any]:
    """Get store performance breakdown by cluster (A/B/C).

    Returns:
        Dict with cluster_revenue_share, cluster_profit_share,
        top_10_stores_revenue_share, bottom_quartile_avg_margin,
        stores_below_breakeven_count.
    """
    try:
        data = _load_data()
        tx = data["transactions"]
        stores = data["stores"]

        tx_stores = tx.merge(
            stores[["store_id", "performance_cluster", "store_format"]],
            on="store_id",
            how="left",
        )

        # Revenue and profit by cluster
        cluster_rev = tx_stores.groupby("performance_cluster")["gross_revenue"].sum()
        cluster_profit = tx_stores.groupby("performance_cluster")["gross_profit"].sum()
        total_rev = cluster_rev.sum()
        total_profit = cluster_profit.sum()

        cluster_revenue_share = {k: round(float(v / total_rev), 4) for k, v in cluster_rev.items()}
        cluster_profit_share = {k: round(float(v / total_profit), 4) for k, v in cluster_profit.items()}

        # Top 10 stores revenue share
        store_rev = tx.groupby("store_id")["gross_revenue"].sum().sort_values(ascending=False)
        top10_share = float(store_rev.head(10).sum() / store_rev.sum())

        # Per-store margin
        store_metrics = (
            tx.groupby("store_id")
            .agg(revenue=("gross_revenue", "sum"), profit=("gross_profit", "sum"))
            .reset_index()
        )
        store_metrics["margin_pct"] = store_metrics["profit"] / store_metrics["revenue"]

        bottom_quartile_threshold = store_metrics["margin_pct"].quantile(0.25)
        bottom_quartile = store_metrics[store_metrics["margin_pct"] <= bottom_quartile_threshold]
        bottom_avg_margin = float(bottom_quartile["margin_pct"].mean())

        stores_below_breakeven = int((store_metrics["margin_pct"] <= 0).sum())

        # Cluster count
        cluster_counts = stores["performance_cluster"].value_counts().to_dict()

        return {
            "cluster_revenue_share": cluster_revenue_share,
            "cluster_profit_share": cluster_profit_share,
            "cluster_store_counts": {k: int(v) for k, v in cluster_counts.items()},
            "top_10_stores_revenue_share": round(top10_share, 4),
            "bottom_quartile_avg_margin_pct": round(bottom_avg_margin * 100, 2),
            "stores_below_breakeven_count": stores_below_breakeven,
            "total_stores": int(stores.shape[0]),
        }
    except Exception as exc:
        logger.error(f"get_store_performance failed: {exc}")
        return {"error": str(exc)}


def get_marketing_roi() -> dict[str, Any]:
    """Get marketing ROI by channel and store format.

    Returns:
        Dict with roi_by_channel, urban_digital_roi, rural_digital_roi,
        urban_tv_roi, rural_tv_roi, optimal_urban_allocation, optimal_rural_allocation.
    """
    try:
        # Use the SimulationEngine's channel ROI table — single source of truth
        from bcgx.simulation.engine import CHANNEL_ROI

        data = _load_data()
        ms = data["marketing_spend"]
        tx = data["transactions"]
        n_months = tx["date"].dt.to_period("M").nunique()

        # Total spend by channel
        channel_spend = ms.groupby("channel")["spend_usd"].sum() / (n_months / 12)
        total_spend = float(channel_spend.sum())

        channel_allocation = {
            ch: round(float(v / total_spend), 4) for ch, v in channel_spend.items()
        }

        # Format spend breakdown
        format_spend = ms.groupby(["store_format", "channel"])["spend_usd"].sum()

        return {
            "roi_by_channel": {
                "urban": CHANNEL_ROI["urban"],
                "rural": CHANNEL_ROI["rural"],
                "suburban": CHANNEL_ROI["suburban"],
            },
            "urban_digital_roi": CHANNEL_ROI["urban"]["digital"],
            "rural_digital_roi": CHANNEL_ROI["rural"]["digital"],
            "urban_tv_roi": CHANNEL_ROI["urban"]["tv"],
            "rural_tv_roi": CHANNEL_ROI["rural"]["tv"],
            "current_channel_allocation": channel_allocation,
            "annual_marketing_spend_usd": round(total_spend, 2),
            "optimal_urban_allocation": {
                "digital": 0.50,
                "email": 0.25,
                "instore_promo": 0.15,
                "tv": 0.05,
                "print": 0.05,
            },
            "optimal_rural_allocation": {
                "tv": 0.40,
                "print": 0.25,
                "instore_promo": 0.20,
                "email": 0.10,
                "digital": 0.05,
            },
            "key_insight": (
                "Urban stores see 3.0x ROI on digital vs 0.8x on TV. "
                "Rural stores see 2.5x ROI on TV vs 0.8x on digital. "
                "Reallocating to format-optimal channels is the highest-ROI marketing action."
            ),
        }
    except Exception as exc:
        logger.error(f"get_marketing_roi failed: {exc}")
        return {"error": str(exc)}


def get_pricing_analysis() -> dict[str, Any]:
    """Get price elasticity and pricing opportunities.

    Returns:
        Dict with elasticity_by_category, private_label_margin_mult,
        inelastic_revenue_opportunity, over_discounted_categories.
    """
    try:
        from bcgx.simulation.engine import PRICE_ELASTICITY

        data = _load_data()
        tx = data["transactions"]
        products = data["products"]

        # Join transactions with products
        prod_tx = tx.merge(
            products[["product_id", "brand_type", "category", "gross_margin_pct"]],
            on="product_id",
            how="left",
        )

        # Margin by brand type
        pl_rev = prod_tx[prod_tx["brand_type"] == "private_label"]["gross_revenue"].sum()
        pl_profit = prod_tx[prod_tx["brand_type"] == "private_label"]["gross_profit"].sum()
        nb_rev = prod_tx[prod_tx["brand_type"] == "national_brand"]["gross_revenue"].sum()
        nb_profit = prod_tx[prod_tx["brand_type"] == "national_brand"]["gross_profit"].sum()

        pl_margin = float(pl_profit / pl_rev) if pl_rev > 0 else 0.0
        nb_margin = float(nb_profit / nb_rev) if nb_rev > 0 else 0.0

        # Inelastic categories (|elasticity| < 1.0)
        inelastic_cats = {k: v for k, v in PRICE_ELASTICITY.items() if abs(v) < 1.0}

        # Revenue opportunity from 5% price increase on inelastic categories
        inelastic_rev_opp = 0.0
        n_months = tx["date"].dt.to_period("M").nunique()
        for cat, elast in inelastic_cats.items():
            cat_prods = products[products["category"] == cat]["product_id"]
            cat_rev = (
                tx[tx["product_id"].isin(cat_prods)]["gross_revenue"].sum()
                / (n_months / 12)
            )
            # Revenue change = cat_rev * (1+0.05) * (1 + elast*0.05) - cat_rev
            rev_change = cat_rev * ((1.05) * (1 + elast * 0.05) - 1)
            inelastic_rev_opp += rev_change

        # Over-discounted categories
        cat_discount = prod_tx.groupby("category")["discount_pct"].mean().sort_values(ascending=False)
        over_discounted = [
            {"category": cat, "avg_discount_pct": round(float(disc) * 100, 2)}
            for cat, disc in cat_discount.head(3).items()
        ]

        # PL mix by revenue
        pl_mix = pl_rev / (pl_rev + nb_rev)

        return {
            "elasticity_by_category": PRICE_ELASTICITY,
            "inelastic_categories": list(inelastic_cats.keys()),
            "elastic_categories": [k for k, v in PRICE_ELASTICITY.items() if abs(v) >= 1.0],
            "private_label_margin_pct": round(pl_margin * 100, 1),
            "national_brand_margin_pct": round(nb_margin * 100, 1),
            "private_label_margin_premium_mult": round(pl_margin / nb_margin, 2) if nb_margin > 0 else None,
            "private_label_revenue_mix_pct": round(float(pl_mix) * 100, 1),
            "inelastic_revenue_opportunity_annual_usd": round(inelastic_rev_opp, 2),
            "over_discounted_categories": over_discounted,
            "key_insight": (
                "Food & Beverage (elasticity -0.45) and Health & Beauty (-0.80) are inelastic — "
                "a 5% price increase would increase revenue. Private label has a "
                f"{round(pl_margin / nb_margin, 2):.2f}x margin advantage over national brands."
            ),
        }
    except Exception as exc:
        logger.error(f"get_pricing_analysis failed: {exc}")
        return {"error": str(exc)}


def get_recommendations() -> dict[str, Any]:
    """Get top strategic recommendations ranked by RICE score.

    Returns:
        Dict with total_recommendations, total_revenue_opportunity_usd, top_3 list.
    """
    try:
        from bcgx.recommendations.engine import RecommendationEngine
        from bcgx.recommendations.prioritizer import RecommendationPrioritizer

        engine = RecommendationEngine()
        recs = engine.generate_all()
        ranked = RecommendationPrioritizer().prioritize(recs)

        total_rev_opp = sum(r.expected_revenue_impact_usd for r in ranked)

        top_3 = [
            {
                "rank": i + 1,
                "title": r.title,
                "category": r.category,
                "priority": r.priority.value,
                "revenue_impact_usd": round(r.expected_revenue_impact_usd, 2),
                "profit_impact_usd": round(r.expected_profit_impact_usd, 2),
                "roi": round(r.roi, 2),
                "timeline": r.timeline.value,
                "rice_score": round(r.rice_score, 1),
                "description": r.description,
                "evidence": r.evidence,
            }
            for i, r in enumerate(ranked[:3])
        ]

        all_recs_summary = [
            {
                "id": r.id,
                "title": r.title,
                "category": r.category,
                "priority": r.priority.value,
                "revenue_impact_usd": round(r.expected_revenue_impact_usd, 2),
                "rice_score": round(r.rice_score, 1),
            }
            for r in ranked
        ]

        return {
            "total_recommendations": len(ranked),
            "total_revenue_opportunity_usd": round(total_rev_opp, 2),
            "top_3": top_3,
            "all_recommendations_summary": all_recs_summary,
        }
    except Exception as exc:
        logger.error(f"get_recommendations failed: {exc}")
        return {"error": str(exc)}


def simulate_scenario(scenario_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Run a business scenario simulation.

    Args:
        scenario_type: One of "price_change", "marketing_reallocation",
                       "churn_reduction", "store_investment".
        parameters: Parameters specific to the scenario type.

    Returns:
        Dict with baseline_revenue, projected_revenue, revenue_delta, etc.
    """
    try:
        from bcgx.simulation.engine import SimulationEngine

        engine = SimulationEngine()
        result = engine.run_scenario_from_dict(scenario_type, parameters)

        return {
            "scenario_type": scenario_type,
            "parameters": parameters,
            "baseline_revenue": round(result.baseline_revenue, 2),
            "projected_revenue": round(result.projected_revenue, 2),
            "revenue_delta": round(result.revenue_delta, 2),
            "revenue_delta_pct": round(result.revenue_delta_pct, 4),
            "baseline_profit": round(result.baseline_profit, 2),
            "projected_profit": round(result.projected_profit, 2),
            "profit_delta": round(result.profit_delta, 2),
            "baseline_margin_pct": round(result.baseline_margin_pct, 2),
            "projected_margin_pct": round(result.projected_margin_pct, 2),
            "margin_delta_pp": round(result.margin_delta_pct, 2),
            "confidence_level": result.confidence_level,
            "timeline_months": result.timeline_months,
            "assumptions": result.assumptions,
            "risks": result.risks,
        }
    except Exception as exc:
        logger.error(f"simulate_scenario failed: {exc}")
        return {"error": str(exc), "scenario_type": scenario_type}


# ── Tool definitions (Anthropic schema) ───────────────────────────────────────

TOOL_DEFINITIONS: list[ToolParam] = [
    {
        "name": "get_financial_summary",
        "description": (
            "Get NovaMart's key financial KPIs including total revenue, total profit, "
            "average gross margin, and the margin trend from the start to the end of the "
            "36-month period. Use this first for any question about overall profitability "
            "or financial health."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_churn_analysis",
        "description": (
            "Get customer churn analysis broken down by loyalty tier (Gold, Silver, Bronze). "
            "Reveals the Silver tier churn acceleration after Month 24, revenue at risk, "
            "and number of churned customers. Use for questions about customer retention, "
            "loyalty program performance, or revenue at risk."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_store_performance",
        "description": (
            "Get store performance breakdown by cluster (A=top, B=mid, C=bottom). "
            "Shows revenue and profit concentration, top-10 store share, and stores "
            "with below-breakeven margins. Use for questions about store prioritisation, "
            "portfolio management, or operational performance."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_marketing_roi",
        "description": (
            "Get marketing ROI by channel (digital, TV, email, print, instore_promo) "
            "and store format (urban, rural, suburban). Shows that urban digital ROI is 3.0x "
            "vs TV at 0.8x, and rural TV ROI is 2.5x vs digital at 0.8x. "
            "Use for questions about marketing budget allocation or channel effectiveness."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_pricing_analysis",
        "description": (
            "Get price elasticity coefficients by category and private label margin advantage. "
            "Identifies inelastic categories (Food & Beverage -0.45, Health & Beauty -0.80) "
            "where price increases would improve revenue. Use for pricing strategy questions "
            "or private label expansion opportunities."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_recommendations",
        "description": (
            "Get all strategic recommendations for NovaMart ranked by RICE score, "
            "with revenue impact, ROI, timeline, and implementation evidence. "
            "Use for questions about what management should do, prioritisation, "
            "or the overall strategic roadmap."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "simulate_scenario",
        "description": (
            "Run a business what-if scenario simulation and get projected revenue, "
            "profit and margin impact with confidence intervals. "
            "Supported scenario types: price_change, marketing_reallocation, "
            "churn_reduction, store_investment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario_type": {
                    "type": "string",
                    "enum": [
                        "price_change",
                        "marketing_reallocation",
                        "churn_reduction",
                        "store_investment",
                    ],
                    "description": "Type of scenario to simulate.",
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "Parameters for the scenario. "
                        "price_change: {category, price_change_pct}. "
                        "marketing_reallocation: {reallocation: {channel: fraction}, store_format?}. "
                        "churn_reduction: {target_segment, churn_reduction_pct, intervention_cost_usd?}. "
                        "store_investment: {n_stores_to_invest, investment_per_store_usd, "
                        "expected_performance_improvement_pct}."
                    ),
                },
            },
            "required": ["scenario_type", "parameters"],
        },
    },
]

# ── Function registry ─────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict[str, Any] = {
    "get_financial_summary": get_financial_summary,
    "get_churn_analysis": get_churn_analysis,
    "get_store_performance": get_store_performance,
    "get_marketing_roi": get_marketing_roi,
    "get_pricing_analysis": get_pricing_analysis,
    "get_recommendations": get_recommendations,
    "simulate_scenario": simulate_scenario,
}
