"""Temporal trend and seasonality analysis for NovaMart metrics.

Business questions answered:
- Is revenue growing or declining — and at what rate?
- Is the margin decline accelerating?
- Which months are seasonal peaks and troughs (critical for inventory and staffing)?
- Is marketing spend becoming more or less efficient over time?

Trend analysis uses OLS regression (scipy.stats.linregress) on monthly time series.
Seasonality is computed via monthly indices (month_avg / year_avg).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class TemporalResult:
    """Result of a temporal trend / seasonality analysis."""

    metric: str
    business_question: str
    trend_direction: str  # "declining" | "growing" | "stable" | "seasonal"
    trend_magnitude: float  # % change from first to last period
    seasonal_peak_month: int | None
    seasonal_trough_month: int | None
    insight: str
    action: str
    monthly_values: dict[str, float]  # {YYYY-MM: value}
    trend_slope: float = field(default=0.0)
    trend_r_squared: float = field(default=0.0)
    trend_p_value: float = field(default=1.0)


class TemporalAnalyzer:
    """Compute time-series trends and seasonal patterns for key NovaMart business metrics.

    All methods return TemporalResult dataclasses; no plotting is performed.
    """

    SIGNIFICANCE_ALPHA: float = 0.05

    # ------------------------------------------------------------------ #
    # Public analysis methods                                              #
    # ------------------------------------------------------------------ #

    def analyze_revenue_trend(self, transactions: pd.DataFrame) -> TemporalResult:
        """Compute monthly revenue trend and seasonality.

        Args:
            transactions: Transaction data with date/year_month and gross_revenue.

        Returns:
            TemporalResult for monthly revenue.
        """
        logger.info("Temporal analysis: revenue trend")

        monthly = self._monthly_sum(transactions, "gross_revenue")
        return self._build_result(
            series=monthly,
            metric="monthly_gross_revenue",
            business_question=(
                "Is NovaMart's revenue growing, declining, or stagnating — and is there a "
                "meaningful seasonal pattern we should plan inventory and staffing around?"
            ),
            growing_insight_template=(
                "NovaMart revenue shows a {direction} trend over the analysis window "
                "(slope=${slope:,.0f}/month, R²={r2:.3f}, p={pval:.4f}), representing a "
                "{magnitude:.1f}% change from the first to the last month in the dataset. "
                "{seasonal_sentence} "
                "While the trend is broadly positive, the growth rate may not outpace rising "
                "operating costs — margin expansion is a separate and more critical challenge."
            ),
            declining_insight_template=(
                "NovaMart revenue is in a {direction} trend (slope=${slope:,.0f}/month, "
                "R²={r2:.3f}, p={pval:.4f}), representing a {magnitude:.1f}% contraction from "
                "the first to the last month. {seasonal_sentence} "
                "This trajectory, if unaddressed, implies meaningful top-line erosion within "
                "the next 12 months and puts pressure on NovaMart's ability to invest in "
                "transformation while managing margin simultaneously."
            ),
            growing_action=(
                "Protect and extend the revenue growth trajectory by reinforcing the highest-"
                "performing store clusters and channels. Build seasonal demand forecasts into "
                "inventory replenishment to avoid stockouts during peak months, which directly "
                "cap revenue upside. Set a quarterly revenue velocity review to detect deceleration "
                "early — a 2-consecutive-month slowdown should trigger a tactical commercial response."
            ),
            declining_action=(
                "Treat the revenue decline as a board-level priority: convene a Revenue Recovery "
                "Task Force within 30 days. Immediate actions include: (1) a store-by-store "
                "diagnostic to identify which locations are driving the decline, (2) a commercial "
                "review of pricing and promotional strategy to identify quick wins, and (3) a "
                "customer retention sprint targeting the highest-value at-risk customers."
            ),
        )

    def analyze_margin_trend(self, transactions: pd.DataFrame) -> TemporalResult:
        """Compute monthly gross margin percentage trend.

        Args:
            transactions: Transaction data with gross_profit and gross_revenue.

        Returns:
            TemporalResult for monthly gross margin %.
        """
        logger.info("Temporal analysis: margin trend")

        tx = transactions.copy()
        tx["year_month"] = tx["year_month"].astype(str)

        monthly_profit = tx.groupby("year_month")["gross_profit"].sum()
        monthly_revenue = tx.groupby("year_month")["gross_revenue"].sum()
        monthly_margin = (monthly_profit / monthly_revenue.clip(lower=1) * 100).sort_index()

        monthly_values = {k: float(v) for k, v in monthly_margin.items()}
        periods = sorted(monthly_values.keys())
        y_vals = np.array([monthly_values[p] for p in periods])
        x_vals = np.arange(len(y_vals), dtype=float)

        slope, intercept, r_value, p_value, _ = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2

        first_val = y_vals[0] if len(y_vals) > 0 else 0.0
        last_val = y_vals[-1] if len(y_vals) > 0 else 0.0
        pct_change = (last_val - first_val) / abs(first_val) * 100 if first_val != 0 else 0.0

        direction = self._classify_direction(slope, p_value)
        peak_m, trough_m = self._seasonal_peaks(monthly_margin)

        is_sig = bool(p_value < self.SIGNIFICANCE_ALPHA)
        mag_str = f"{abs(pct_change):.1f}pp"

        if "declining" in direction:
            insight = (
                f"NovaMart gross margin is on a statistically {'significant' if is_sig else 'non-significant'} "
                f"declining trend (slope={slope:.3f}pp/month, R²={r_squared:.3f}, p={p_value:.4f}). "
                f"Margin has contracted {mag_str} over the analysis window, from {first_val:.1f}% to "
                f"{last_val:.1f}%. This is the most critical financial finding in the EDA: margin "
                "compression of this magnitude — if annualised — will eliminate NovaMart's profitability "
                "within the medium-term planning horizon without structural intervention. The decline "
                "likely reflects a combination of promotional intensification, unfavourable product mix "
                "shift, and input cost inflation not being passed through to prices."
            )
            action = (
                "Declare a Margin Recovery Programme as the primary strategic priority for the next "
                "12 months. Immediate actions: (1) implement a discount depth ceiling by category, "
                "(2) accelerate private-label penetration in top-10 revenue categories, (3) renegotiate "
                "supplier terms for the top-50 SKUs by cost, and (4) implement a quarterly margin "
                "review cadence with store-level accountability. Target: recover 200bps of gross "
                "margin within 12 months through mix shift and promotional discipline."
            )
        else:
            insight = (
                f"Gross margin trend is {direction} (slope={slope:.3f}pp/month, R²={r_squared:.3f}, "
                f"p={p_value:.4f}), with a {mag_str} change from {first_val:.1f}% to {last_val:.1f}% "
                "over the period. Stable or improving margins suggest NovaMart's pricing and product "
                "mix decisions are broadly sound, but should not induce complacency — the competitive "
                "environment can shift margin dynamics rapidly."
            )
            action = (
                "Maintain margin discipline by embedding a monthly margin tracking dashboard into "
                "executive reporting. Identify the categories contributing most to margin improvement "
                "and scale those strategies. Establish a 'margin floor' policy: if monthly margin "
                f"falls below {last_val - 2:.1f}%, an automatic commercial review is triggered."
            )

        return TemporalResult(
            metric="monthly_gross_margin_pct",
            business_question=(
                "Is NovaMart's gross margin declining — and is the rate of decline accelerating "
                "to a level that threatens the business model?"
            ),
            trend_direction=direction,
            trend_magnitude=pct_change,
            seasonal_peak_month=peak_m,
            seasonal_trough_month=trough_m,
            insight=insight,
            action=action,
            monthly_values=monthly_values,
            trend_slope=float(slope),
            trend_r_squared=float(r_squared),
            trend_p_value=float(p_value),
        )

    def analyze_customer_churn_trend(
        self, transactions: pd.DataFrame, customers: pd.DataFrame
    ) -> TemporalResult:
        """Compute monthly trend in new vs. returning customer ratio (proxy for churn).

        Churn proxy: fraction of purchasing customers in a given month who have
        not purchased in the preceding 3 months.

        Args:
            transactions: Transaction data.
            customers: Customer master (for segment context).

        Returns:
            TemporalResult for customer activity / churn proxy.
        """
        logger.info("Temporal analysis: customer churn trend")

        tx = transactions[["customer_id", "year_month", "date"]].copy()
        tx["year_month"] = tx["year_month"].astype(str)

        # For each month, compute unique active customers
        monthly_active = (
            tx.groupby("year_month")["customer_id"].nunique().sort_index()
        )
        monthly_values = {k: float(v) for k, v in monthly_active.items()}
        periods = sorted(monthly_values.keys())
        y_vals = np.array([monthly_values[p] for p in periods])
        x_vals = np.arange(len(y_vals), dtype=float)

        slope, intercept, r_value, p_value, _ = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2

        first_val = y_vals[0] if len(y_vals) > 0 else 1.0
        last_val = y_vals[-1] if len(y_vals) > 0 else 1.0
        pct_change = (last_val - first_val) / abs(first_val) * 100 if first_val != 0 else 0.0

        direction = self._classify_direction(slope, p_value)
        peak_m, trough_m = self._seasonal_peaks(monthly_active)

        is_sig = bool(p_value < self.SIGNIFICANCE_ALPHA)

        if "declining" in direction:
            insight = (
                f"Monthly active customer counts show a statistically {'significant' if is_sig else 'non-significant'} "
                f"declining trend (slope={slope:.1f} customers/month, R²={r_squared:.3f}, p={p_value:.4f}). "
                f"Active customers have fallen {abs(pct_change):.1f}% from {first_val:.0f} to {last_val:.0f} "
                "per month over the analysis window. This is a leading indicator of revenue risk: declining "
                "customer engagement precedes revenue decline by 2-3 months and, once established, is "
                "significantly more expensive to reverse than to prevent."
            )
            action = (
                "Launch an immediate churn intervention programme: (1) identify customers with no "
                "purchase in the last 60 days who previously purchased monthly, (2) deploy a "
                "win-back campaign with personalised offers based on historical purchase patterns, "
                "(3) add a monthly 'Customer Health Score' KPI to executive dashboards. Target: "
                "reduce monthly customer decline rate by 50% within two quarters."
            )
        else:
            insight = (
                f"Monthly active customer counts are {direction} (slope={slope:.1f}/month, "
                f"R²={r_squared:.3f}, p={p_value:.4f}). Stable or growing engagement suggests "
                "NovaMart's customer base is broadly retained, though growth in active customers "
                "does not necessarily indicate healthy underlying churn — new customer acquisition "
                "may be masking elevated attrition in the existing base."
            )
            action = (
                "Decompose active customer growth into new vs. returning customer cohorts monthly. "
                "If new customer growth is the primary driver, accelerate retention programmes to "
                "extend average customer lifetime. A 10% improvement in customer retention rate "
                "typically drives 25-95% improvement in customer lifetime value."
            )

        return TemporalResult(
            metric="monthly_active_customers",
            business_question=(
                "Is NovaMart's active customer base growing or contracting — and is churn "
                "accelerating following the loyalty programme changes?"
            ),
            trend_direction=direction,
            trend_magnitude=pct_change,
            seasonal_peak_month=peak_m,
            seasonal_trough_month=trough_m,
            insight=insight,
            action=action,
            monthly_values=monthly_values,
            trend_slope=float(slope),
            trend_r_squared=float(r_squared),
            trend_p_value=float(p_value),
        )

    def analyze_marketing_efficiency_trend(
        self, transactions: pd.DataFrame, marketing: pd.DataFrame
    ) -> TemporalResult:
        """Compute monthly revenue-per-marketing-dollar (efficiency) trend.

        Args:
            transactions: Transaction data.
            marketing: Marketing spend data.

        Returns:
            TemporalResult for marketing efficiency.
        """
        logger.info("Temporal analysis: marketing efficiency trend")

        monthly_rev = (
            transactions.groupby("year_month")["gross_revenue"]
            .sum()
            .reset_index(name="revenue")
        )
        monthly_mkt = (
            marketing.groupby("year_month")["spend_usd"]
            .sum()
            .reset_index(name="spend")
        )
        merged = monthly_rev.merge(monthly_mkt, on="year_month", how="inner")
        merged["efficiency"] = merged["revenue"] / merged["spend"].clip(lower=1)
        merged = merged.sort_values("year_month")

        efficiency_series = merged.set_index("year_month")["efficiency"]
        monthly_values = {k: float(v) for k, v in efficiency_series.items()}
        periods = sorted(monthly_values.keys())
        y_vals = np.array([monthly_values[p] for p in periods])
        x_vals = np.arange(len(y_vals), dtype=float)

        slope, intercept, r_value, p_value, _ = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2

        first_val = y_vals[0] if len(y_vals) > 0 else 1.0
        last_val = y_vals[-1] if len(y_vals) > 0 else 1.0
        pct_change = (last_val - first_val) / abs(first_val) * 100 if first_val != 0 else 0.0

        direction = self._classify_direction(slope, p_value)
        peak_m, trough_m = self._seasonal_peaks(efficiency_series)
        avg_efficiency = float(efficiency_series.mean())

        is_sig = bool(p_value < self.SIGNIFICANCE_ALPHA)

        if "declining" in direction:
            insight = (
                f"Marketing efficiency (revenue per $1 of marketing spend) is declining "
                f"({'significantly' if is_sig else 'non-significantly'}) over time "
                f"(slope={slope:.3f} revenue/$/month, R²={r_squared:.3f}, p={p_value:.4f}). "
                f"On average, each $1 of marketing generates ${avg_efficiency:.2f} in revenue, "
                f"but this ratio has deteriorated {abs(pct_change):.1f}% since the start of the "
                "analysis window. Declining marketing efficiency means NovaMart is spending more "
                "to generate the same (or less) revenue — a structural profitability risk as "
                "competitive pressure intensifies."
            )
            action = (
                "Immediately audit the marketing channel mix for ROI contribution. Reallocate "
                "budget away from declining-efficiency channels toward proven performers. "
                "Introduce a minimum marketing ROI threshold (e.g., $3 revenue per $1 spend) "
                "as a budget gate. Invest in closed-loop measurement infrastructure — the "
                "inability to measure attribution is a primary driver of inefficient spend."
            )
        else:
            insight = (
                f"Marketing efficiency is {direction} (slope={slope:.3f}/month, R²={r_squared:.3f}), "
                f"with an average return of ${avg_efficiency:.2f} per $1 spent. "
                "Stable or improving marketing efficiency suggests the current channel mix "
                "and targeting strategy is broadly effective, though there is typically "
                "significant headroom for further optimisation through personalisation and "
                "programmatic budget allocation."
            )
            action = (
                "Scale what is working: identify the top-3 marketing channels by ROI and "
                "increase their budget allocation by 20% while correspondingly reducing spend "
                "on the bottom-3 channels. Set a quarterly marketing efficiency review to "
                "ensure the trend is maintained as the portfolio scales."
            )

        return TemporalResult(
            metric="marketing_revenue_efficiency",
            business_question=(
                "Is NovaMart's marketing spend becoming more or less productive over time, "
                "and which months represent peak and trough marketing ROI?"
            ),
            trend_direction=direction,
            trend_magnitude=pct_change,
            seasonal_peak_month=peak_m,
            seasonal_trough_month=trough_m,
            insight=insight,
            action=action,
            monthly_values=monthly_values,
            trend_slope=float(slope),
            trend_r_squared=float(r_squared),
            trend_p_value=float(p_value),
        )

    def run_all(self, data: dict[str, pd.DataFrame]) -> list[TemporalResult]:
        """Execute all temporal analyses.

        Args:
            data: Dict produced by DataLoader.load_all().

        Returns:
            List of TemporalResult objects.
        """
        logger.info("Running all temporal analyses")
        results: list[TemporalResult] = [
            self.analyze_revenue_trend(data["transactions"]),
            self.analyze_margin_trend(data["transactions"]),
            self.analyze_customer_churn_trend(data["transactions"], data["customers"]),
            self.analyze_marketing_efficiency_trend(data["transactions"], data["marketing_spend"]),
        ]
        logger.success(f"Temporal analysis complete: {len(results)} results generated")
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _monthly_sum(transactions: pd.DataFrame, col: str) -> pd.Series:  # type: ignore[type-arg]
        """Group transactions by year_month and sum the given column."""
        return (
            transactions.groupby("year_month")[col].sum().sort_index()
        )

    def _build_result(
        self,
        series: pd.Series,  # type: ignore[type-arg]
        metric: str,
        business_question: str,
        growing_insight_template: str,
        declining_insight_template: str,
        growing_action: str,
        declining_action: str,
    ) -> TemporalResult:
        """Fit OLS trend, classify direction, compute seasonality, build TemporalResult."""
        monthly_values = {k: float(v) for k, v in series.items()}
        periods = sorted(monthly_values.keys())
        y_vals = np.array([monthly_values[p] for p in periods])
        x_vals = np.arange(len(y_vals), dtype=float)

        slope, intercept, r_value, p_value, _ = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2

        first_val = y_vals[0] if len(y_vals) > 0 else 1.0
        last_val = y_vals[-1] if len(y_vals) > 0 else 1.0
        pct_change = (last_val - first_val) / abs(first_val) * 100 if first_val != 0 else 0.0

        direction = self._classify_direction(slope, p_value)
        peak_m, trough_m = self._seasonal_peaks(series)

        seasonal_sentence = ""
        if peak_m and trough_m:
            seasonal_sentence = (
                f"Seasonality is evident: the index peaks in month {peak_m} "
                f"and troughs in month {trough_m}, a pattern that should drive inventory "
                "planning and staffing decisions."
            )

        template = growing_insight_template if "growing" in direction or "stable" in direction else declining_insight_template
        insight = template.format(
            direction=direction,
            slope=slope,
            r2=r_squared,
            pval=p_value,
            magnitude=abs(pct_change),
            seasonal_sentence=seasonal_sentence,
        )
        action = growing_action if "growing" in direction or "stable" in direction else declining_action

        return TemporalResult(
            metric=metric,
            business_question=business_question,
            trend_direction=direction,
            trend_magnitude=pct_change,
            seasonal_peak_month=peak_m,
            seasonal_trough_month=trough_m,
            insight=insight,
            action=action,
            monthly_values=monthly_values,
            trend_slope=float(slope),
            trend_r_squared=float(r_squared),
            trend_p_value=float(p_value),
        )

    def _classify_direction(self, slope: float, p_value: float) -> str:
        """Classify trend direction using slope sign and statistical significance."""
        is_significant = p_value < self.SIGNIFICANCE_ALPHA
        if not is_significant:
            return "stable"
        return "growing" if slope > 0 else "declining"

    @staticmethod
    def _seasonal_peaks(series: pd.Series) -> tuple[int | None, int | None]:  # type: ignore[type-arg]
        """Compute seasonal peak and trough month using monthly index method.

        Monthly index = month_avg / year_avg. Months with index > 1 are above-average.

        Args:
            series: Time series indexed by YYYY-MM strings.

        Returns:
            (peak_month_int, trough_month_int) or (None, None) if insufficient data.
        """
        if len(series) < 12:
            return None, None

        df = pd.DataFrame({"value": series})
        df.index = pd.to_datetime(df.index.astype(str) + "-01")
        df["month"] = df.index.month
        df["year"] = df.index.year

        year_avg = df.groupby("year")["value"].mean()
        df["year_avg"] = df["year"].map(year_avg)
        df["monthly_index"] = df["value"] / df["year_avg"].clip(lower=1e-9)

        monthly_index = df.groupby("month")["monthly_index"].mean()
        if monthly_index.empty:
            return None, None

        peak_m = int(monthly_index.idxmax())
        trough_m = int(monthly_index.idxmin())
        return peak_m, trough_m
