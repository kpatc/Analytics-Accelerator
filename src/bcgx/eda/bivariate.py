"""Bivariate relationship analysis for NovaMart key metric pairs.

Business questions answered:
- Does marketing spend drive revenue?
- Do larger stores outperform smaller ones?
- Is customer loyalty tier correlated with spend?
- Does heavier discounting erode margins?

All relationships are tested with appropriate correlation statistics, p-values, and
translated into business recommendations for NovaMart leadership.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class BivariateResult:
    """Result of a bivariate correlation / association analysis."""

    x_var: str
    y_var: str
    business_question: str
    correlation: float
    correlation_type: str  # "pearson" | "spearman" | "eta_squared"
    p_value: float
    is_significant: bool  # True when p < 0.05
    insight: str
    action: str


class BivariateAnalyzer:
    """Test pairwise relationships between business metrics.

    Chooses between Pearson (continuous × continuous, approximately normal) and
    Spearman (ordinal or skewed) depending on the data context.  No plotting is
    performed; results are pure data structures.
    """

    ALPHA: float = 0.05

    # ------------------------------------------------------------------ #
    # Public analysis methods                                              #
    # ------------------------------------------------------------------ #

    def analyze_marketing_revenue(
        self,
        transactions: pd.DataFrame,
        marketing: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> BivariateResult:
        """Test whether monthly marketing spend predicts monthly store revenue.

        Uses Spearman correlation to handle the right-skewed spend distributions.

        Args:
            transactions: Transaction data with year_month, store_id, gross_revenue.
            marketing: Marketing spend data with year_month, store_id, spend_usd.
            stores: Store master (not used directly but passed for API consistency).

        Returns:
            BivariateResult for marketing spend → revenue.
        """
        logger.info("Bivariate analysis: marketing spend vs revenue")

        monthly_rev = (
            transactions.groupby(["store_id", "year_month"])["gross_revenue"]
            .sum()
            .reset_index(name="revenue")
        )
        monthly_mkt = (
            marketing.groupby(["store_id", "year_month"])["spend_usd"]
            .sum()
            .reset_index(name="marketing_spend")
        )
        merged = monthly_rev.merge(monthly_mkt, on=["store_id", "year_month"], how="inner").dropna()

        corr, p_val = stats.spearmanr(merged["marketing_spend"], merged["revenue"])
        is_sig = bool(p_val < self.ALPHA)

        direction = "positive" if corr > 0 else "negative"
        strength = self._correlation_strength(abs(corr))

        if is_sig:
            insight = (
                f"Marketing spend exhibits a {strength} {direction} Spearman correlation with "
                f"monthly store revenue (ρ={corr:.3f}, p={p_val:.4f}). While correlation does not "
                f"establish causality, the relationship suggests marketing investment is broadly "
                f"associated with revenue outcomes across the portfolio. However, ρ={corr:.3f} "
                f"implies {corr**2 * 100:.1f}% of revenue variance is explained by spend levels "
                "alone — indicating that channel mix, store format, and local market conditions "
                "account for the majority of revenue variability."
            )
            action = (
                "Shift from 'spend more' to 'spend smarter': build a marketing mix model to "
                "decompose the revenue attribution across channels. Prioritise budget reallocation "
                "from low-ROI channels (TV/print) to high-ROI digital channels before increasing "
                "total spend. Test a 20% budget reallocation in 50 pilot stores to measure "
                "incremental revenue lift before full-portfolio rollout."
            )
        else:
            insight = (
                f"Marketing spend shows no statistically significant correlation with monthly "
                f"revenue (ρ={corr:.3f}, p={p_val:.4f}), suggesting that current spend levels "
                "and channel allocations are not effectively driving revenue. This is a red flag: "
                "NovaMart may be deploying marketing budget without measurable return."
            )
            action = (
                "Halt automatic marketing budget renewals and require each channel to demonstrate "
                "measurable revenue attribution. Commission a marketing effectiveness audit within "
                "30 days, with a mandate to reallocate 30% of spend to channels with proven ROI "
                "within 90 days."
            )

        return BivariateResult(
            x_var="marketing_spend_usd",
            y_var="monthly_revenue",
            business_question="Does marketing spend drive revenue, and is the relationship strong enough to justify current budget levels?",
            correlation=float(corr),
            correlation_type="spearman",
            p_value=float(p_val),
            is_significant=is_sig,
            insight=insight,
            action=action,
        )

    def analyze_store_size_performance(
        self, transactions: pd.DataFrame, stores: pd.DataFrame
    ) -> BivariateResult:
        """Test whether store square footage predicts total revenue.

        Args:
            transactions: Transaction data.
            stores: Store master with sq_footage.

        Returns:
            BivariateResult for store size → revenue.
        """
        logger.info("Bivariate analysis: store size vs revenue")

        store_rev = (
            transactions.groupby("store_id")["gross_revenue"]
            .sum()
            .reset_index(name="total_revenue")
        )
        merged = store_rev.merge(stores[["store_id", "sq_footage"]], on="store_id", how="inner").dropna()

        corr, p_val = stats.pearsonr(merged["sq_footage"], merged["total_revenue"])
        is_sig = bool(p_val < self.ALPHA)
        strength = self._correlation_strength(abs(corr))

        revenue_per_sqft = (
            merged["total_revenue"] / merged["sq_footage"]
        )
        avg_rev_per_sqft = revenue_per_sqft.mean()

        if is_sig:
            insight = (
                f"Store size (sq_footage) has a {strength} Pearson correlation with total revenue "
                f"(r={corr:.3f}, p={p_val:.4f}), explaining {corr**2 * 100:.1f}% of revenue variance. "
                f"Average revenue productivity is ${avg_rev_per_sqft:.2f} per sq ft annually. "
                "While larger stores generate more absolute revenue, the question is whether they "
                "generate proportionally more profit — large-format stores carry higher fixed costs "
                "(rent, labor) that can compress net margins even when top-line performance appears strong."
            )
            action = (
                "Conduct a revenue-per-sq-ft analysis by store format to identify underperforming "
                "large-format locations. For stores in the bottom quartile of revenue/sq-ft, evaluate "
                "whether right-sizing (subletting excess space, converting to mixed-use) or targeted "
                "category expansion can improve productivity. Set a minimum revenue-per-sq-ft "
                f"threshold of ${avg_rev_per_sqft * 0.75:.2f} as a store performance KPI."
            )
        else:
            insight = (
                f"Store size shows no significant linear relationship with revenue "
                f"(r={corr:.3f}, p={p_val:.4f}), suggesting that location, catchment demographics, "
                "and management quality matter more than physical scale. This challenges the assumption "
                "that larger format stores are inherently better investments."
            )
            action = (
                "Deprioritise store size as a site-selection criterion. Instead, build a predictive "
                "site-selection model that incorporates catchment demographics, competitor density, "
                "and traffic patterns — factors that appear more predictive of revenue than square footage."
            )

        return BivariateResult(
            x_var="sq_footage",
            y_var="total_revenue",
            business_question="Do larger stores generate proportionally higher revenue, justifying the premium rent and operating costs of large-format locations?",
            correlation=float(corr),
            correlation_type="pearson",
            p_value=float(p_val),
            is_significant=is_sig,
            insight=insight,
            action=action,
        )

    def analyze_loyalty_tier_spend(
        self, transactions: pd.DataFrame, customers: pd.DataFrame
    ) -> BivariateResult:
        """Test whether loyalty tier is associated with customer total spend.

        Uses Spearman rank correlation after encoding tiers ordinally.

        Args:
            transactions: Transaction data.
            customers: Customer master with loyalty_tier.

        Returns:
            BivariateResult for loyalty tier → spend.
        """
        logger.info("Bivariate analysis: loyalty tier vs customer spend")

        cust_spend = (
            transactions.groupby("customer_id")["gross_revenue"]
            .sum()
            .reset_index(name="total_spend")
        )
        merged = cust_spend.merge(
            customers[["customer_id", "loyalty_tier"]], on="customer_id", how="inner"
        ).dropna()

        tier_map = {"Bronze": 1, "Silver": 2, "Gold": 3}
        merged["tier_rank"] = merged["loyalty_tier"].map(tier_map).fillna(0)

        valid = merged[merged["tier_rank"] > 0]
        corr, p_val = stats.spearmanr(valid["tier_rank"], valid["total_spend"])
        is_sig = bool(p_val < self.ALPHA)
        strength = self._correlation_strength(abs(corr))

        tier_avg = merged.groupby("loyalty_tier")["total_spend"].mean()
        gold_avg = tier_avg.get("Gold", 0.0)
        bronze_avg = tier_avg.get("Bronze", 1.0)
        spend_ratio = gold_avg / bronze_avg if bronze_avg > 0 else 1.0

        if is_sig:
            insight = (
                f"Loyalty tier has a {strength} positive Spearman correlation with customer spend "
                f"(ρ={corr:.3f}, p={p_val:.4f}). Gold-tier customers spend {spend_ratio:.1f}× "
                f"more than Bronze-tier on average (${gold_avg:.0f} vs ${bronze_avg:.0f}). "
                "This confirms that the loyalty programme is stratifying the customer base by value — "
                "but causality is ambiguous: high spenders may select into Gold tier rather than Gold "
                "tier causing higher spend. The Silver-tier disruption (loyalty fee increase) may "
                "be particularly damaging if Silver members represent a high-growth, ascending cohort."
            )
            action = (
                "Design a Silver-to-Gold conversion funnel: identify Silver customers within 20% "
                "of Gold spend threshold and offer targeted promotions to bridge the gap. Protect "
                "the Gold tier from any loyalty programme changes — their spend contribution is "
                "disproportionately high and their switching cost is lower than commonly assumed. "
                "Monitor Silver-tier spend velocity monthly and alert when MoM decline exceeds 10%."
            )
        else:
            insight = (
                f"Loyalty tier shows no statistically significant association with customer spend "
                f"(ρ={corr:.3f}, p={p_val:.4f}), suggesting the loyalty programme is not "
                "effectively differentiating customer behaviour by tier. This is a fundamental "
                "design failure: a loyalty programme should drive measurable spend lift."
            )
            action = (
                "Commission a loyalty programme effectiveness review. The programme is not "
                "achieving its primary objective of spend differentiation. Consider a complete "
                "redesign focused on reward structures that incentivise incremental spend rather "
                "than simply rewarding historical behaviour."
            )

        return BivariateResult(
            x_var="loyalty_tier",
            y_var="customer_total_spend",
            business_question="Is the loyalty programme effectively stratifying customers by spend, and is tier membership a reliable predictor of customer value?",
            correlation=float(corr),
            correlation_type="spearman",
            p_value=float(p_val),
            is_significant=is_sig,
            insight=insight,
            action=action,
        )

    def analyze_discount_margin(self, transactions: pd.DataFrame) -> BivariateResult:
        """Test whether higher discount rates erode gross margins.

        Args:
            transactions: Transaction data with discount_pct and gross_profit/gross_revenue.

        Returns:
            BivariateResult for discount rate → gross margin.
        """
        logger.info("Bivariate analysis: discount rate vs gross margin")

        tx = transactions.copy()
        tx["margin_pct"] = np.where(
            tx["gross_revenue"] > 0,
            tx["gross_profit"] / tx["gross_revenue"] * 100,
            np.nan,
        )
        tx = tx.dropna(subset=["margin_pct", "discount_pct"])

        # Sample for performance if very large
        if len(tx) > 100_000:
            tx = tx.sample(n=100_000, random_state=42)

        corr, p_val = stats.pearsonr(tx["discount_pct"], tx["margin_pct"])
        is_sig = bool(p_val < self.ALPHA)

        avg_discount = tx["discount_pct"].mean() * 100
        high_disc_mask = tx["discount_pct"] > tx["discount_pct"].quantile(0.75)
        low_disc_mask = tx["discount_pct"] <= tx["discount_pct"].quantile(0.25)
        high_disc_margin = tx.loc[high_disc_mask, "margin_pct"].mean()
        low_disc_margin = tx.loc[low_disc_mask, "margin_pct"].mean()
        margin_erosion = high_disc_margin - low_disc_margin

        if is_sig:
            insight = (
                f"Discount rate has a statistically significant correlation with gross margin "
                f"(r={corr:.3f}, p={p_val:.4f}). Transactions in the top quartile of discount "
                f"depth average {high_disc_margin:.1f}% gross margin, versus {low_disc_margin:.1f}% "
                f"for the bottom quartile — a {abs(margin_erosion):.1f}pp margin erosion from "
                f"heavy discounting. With an average portfolio discount of {avg_discount:.1f}%, "
                "NovaMart is trading significant margin points for volume that may not be "
                "incrementally profitable once cost-to-serve is considered."
            )
            action = (
                "Implement a discount optimisation model: each category and channel should have a "
                "maximum discount threshold that preserves the minimum acceptable gross margin. "
                "Move to personalised discount targeting (highest discounts only for price-sensitive "
                "customers who would not purchase at full price), and eliminate blanket promotional "
                f"events. A 5pp reduction in average discount depth would improve gross margin by "
                f"approximately {abs(corr) * 5:.1f}pp — equivalent to meaningful profit recovery "
                "without requiring volume growth."
            )
        else:
            insight = (
                f"The correlation between discount rate and gross margin is not statistically "
                f"significant (r={corr:.3f}, p={p_val:.4f}). This may indicate that NovaMart's "
                "discount policy is already well-calibrated by product and category, or that "
                "other factors (product mix, volume effects) are masking the true relationship."
            )
            action = (
                "Conduct a category-level discount effectiveness analysis — the aggregate signal "
                "may be masking important category-specific patterns where heavy discounting is "
                "deeply margin-dilutive. Do not interpret the lack of aggregate significance as "
                "permission to discount more aggressively."
            )

        return BivariateResult(
            x_var="discount_pct",
            y_var="gross_margin_pct",
            business_question="How much gross margin is being sacrificed through discounting, and is the volume traded worth the margin cost?",
            correlation=float(corr),
            correlation_type="pearson",
            p_value=float(p_val),
            is_significant=is_sig,
            insight=insight,
            action=action,
        )

    def run_all(self, data: dict[str, pd.DataFrame]) -> list[BivariateResult]:
        """Execute all bivariate analyses.

        Args:
            data: Dict produced by DataLoader.load_all().

        Returns:
            List of BivariateResult objects.
        """
        logger.info("Running all bivariate analyses")
        results: list[BivariateResult] = [
            self.analyze_marketing_revenue(data["transactions"], data["marketing_spend"], data["stores"]),
            self.analyze_store_size_performance(data["transactions"], data["stores"]),
            self.analyze_loyalty_tier_spend(data["transactions"], data["customers"]),
            self.analyze_discount_margin(data["transactions"]),
        ]
        logger.success(f"Bivariate analysis complete: {len(results)} results generated")
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _correlation_strength(abs_corr: float) -> str:
        """Classify correlation magnitude using Cohen (1988) thresholds."""
        if abs_corr >= 0.70:
            return "strong"
        elif abs_corr >= 0.40:
            return "moderate"
        elif abs_corr >= 0.20:
            return "weak"
        else:
            return "negligible"
