"""Financial impact calculator for NovaMart scenario simulations.

Computes NPV, payback period, ROI and annualised impact for any scenario output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FinancialImpact:
    """Full financial impact summary for a business initiative.

    Attributes:
        revenue_impact_usd: Total projected revenue change in USD.
        profit_impact_usd: Total projected profit change in USD.
        roi: Return on investment = profit_impact / implementation_cost.
        payback_months: Months until cumulative profit covers implementation cost.
        npv_3yr: Net present value over 3 years at the given discount rate.
        annualized_impact: Average annual profit impact in USD.
    """

    revenue_impact_usd: float
    profit_impact_usd: float
    roi: float  # profit_impact / implementation_cost
    payback_months: int
    npv_3yr: float  # simple NPV at discount_rate
    annualized_impact: float


def calculate_financial_impact(
    revenue_delta: float,
    profit_delta: float,
    implementation_cost: float,
    timeline_months: int,
    discount_rate: float = 0.10,
) -> FinancialImpact:
    """Compute financial impact metrics for a scenario simulation.

    Args:
        revenue_delta: Annual revenue change in USD (positive = gain).
        profit_delta: Annual profit change in USD (positive = gain).
        implementation_cost: One-time cost to implement the initiative in USD.
        timeline_months: Expected implementation timeline in months.
        discount_rate: Annual discount rate for NPV calculation (default 10%).

    Returns:
        FinancialImpact with ROI, payback, NPV and annualised impact.

    Example::

        impact = calculate_financial_impact(
            revenue_delta=5_000_000,
            profit_delta=1_500_000,
            implementation_cost=500_000,
            timeline_months=6,
        )
        print(f"ROI: {impact.roi:.1f}x, Payback: {impact.payback_months} months")
    """
    # ROI = net profit impact / cost (return multiple, not percentage)
    if implementation_cost > 0:
        roi = profit_delta / implementation_cost
    else:
        roi = float("inf") if profit_delta > 0 else 0.0

    # Payback: months to recover implementation_cost from monthly profit gains
    monthly_profit_gain = profit_delta / 12.0
    if monthly_profit_gain > 0 and implementation_cost > 0:
        payback_months = int(implementation_cost / monthly_profit_gain)
    elif implementation_cost <= 0:
        payback_months = 0
    else:
        payback_months = 999  # never breaks even

    # 3-year NPV at annual discount_rate
    # Simple model: profit_delta is annual and starts after timeline_months
    ramp_year = timeline_months / 12.0
    npv_3yr = 0.0
    for year in range(1, 4):
        t = year + ramp_year  # time from today (years)
        discounted = profit_delta / ((1 + discount_rate) ** t)
        npv_3yr += discounted
    npv_3yr -= implementation_cost  # subtract upfront cost

    # Annualised impact: average over 3 years including ramp period
    annualized_impact = (profit_delta * 3 - implementation_cost) / 3

    return FinancialImpact(
        revenue_impact_usd=revenue_delta,
        profit_impact_usd=profit_delta,
        roi=roi,
        payback_months=min(payback_months, 999),
        npv_3yr=npv_3yr,
        annualized_impact=annualized_impact,
    )
