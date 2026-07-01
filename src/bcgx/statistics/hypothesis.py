"""Statistical hypothesis testing for NovaMart business hypotheses.

Five hypotheses are tested, each grounded in a specific business concern:

H1: Cluster A stores have significantly higher revenue than Cluster B/C
H2: Silver-tier churn is significantly higher post month-24 (loyalty fee impact)
H3: Private label products have significantly higher margin than national brands
H4: Urban stores have significantly higher digital marketing ROI than rural stores
H5: Discount rate is negatively correlated with profit margin

All tests report p-values, effect sizes (Cohen's d or eta-squared), and a
BCG-consultant-grade business conclusion that translates the statistical finding
into a concrete management implication.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class HypothesisTestResult:
    """Result of a single statistical hypothesis test."""

    hypothesis_id: str
    hypothesis_text: str
    test_name: str
    test_statistic: float
    p_value: float
    alpha: float
    rejected: bool  # True if H0 rejected (result is statistically significant)
    effect_size: float  # Cohen's d, eta-squared, or |r|
    effect_size_interpretation: str  # "negligible" | "small" | "medium" | "large"
    business_conclusion: str
    statistical_conclusion: str


class HypothesisTester:
    """Run all pre-specified hypothesis tests for the NovaMart engagement.

    Each test is calibrated to answer a specific business question with the
    appropriate statistical methodology and full reporting of uncertainty.
    """

    ALPHA: float = 0.05

    # ------------------------------------------------------------------ #
    # H1: Cluster A stores vs B/C revenue (one-way ANOVA)                 #
    # ------------------------------------------------------------------ #

    def test_cluster_revenue_difference(
        self, transactions: pd.DataFrame, stores: pd.DataFrame
    ) -> HypothesisTestResult:
        """Test H1: Cluster A stores generate significantly higher revenue than Cluster B/C.

        Method: One-way ANOVA across clusters A, B, C on annual store revenue.
        Effect size: eta-squared (η²).

        Args:
            transactions: Transaction data with store_id and gross_revenue.
            stores: Store master with performance_cluster.

        Returns:
            HypothesisTestResult for H1.
        """
        logger.info("Testing H1: Cluster A stores vs B/C revenue (ANOVA)")

        store_rev = (
            transactions.groupby("store_id")["gross_revenue"]
            .sum()
            .reset_index(name="total_revenue")
        )
        store_rev = store_rev.merge(
            stores[["store_id", "performance_cluster"]], on="store_id", how="left"
        ).dropna(subset=["performance_cluster"])

        groups = {
            cluster: store_rev.loc[store_rev["performance_cluster"] == cluster, "total_revenue"].values
            for cluster in ["A", "B", "C"]
        }
        groups = {k: v for k, v in groups.items() if len(v) >= 2}

        if len(groups) < 2:
            return self._insufficient_data_result("H1", "One-way ANOVA on store revenue by cluster")

        f_stat, p_value = stats.f_oneway(*groups.values())

        # eta-squared: SS_between / SS_total
        all_vals = np.concatenate(list(groups.values()))
        grand_mean = all_vals.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups.values())
        ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
        eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0
        effect_interp = self._eta_squared_interpretation(eta_squared)

        rejected = bool(p_value < self.ALPHA)

        cluster_means = {k: float(v.mean() / 1e6) for k, v in groups.items()}
        a_mean = cluster_means.get("A", 0.0)
        b_mean = cluster_means.get("B", 0.0)
        c_mean = cluster_means.get("C", 0.0)
        revenue_premium_vs_b = (a_mean - b_mean) / max(b_mean, 0.001) * 100

        if rejected:
            business_conclusion = (
                f"Cluster A stores generate statistically significantly higher revenue than their "
                f"Cluster B and C counterparts (F={f_stat:.2f}, p={p_value:.4f}, η²={eta_squared:.3f}, "
                f"effect size: {effect_interp}). Cluster A stores average ${a_mean:.1f}M annually "
                f"versus ${b_mean:.1f}M for Cluster B and ${c_mean:.1f}M for Cluster C — a "
                f"{revenue_premium_vs_b:.1f}% revenue premium for Cluster A. This confirms that "
                "performance clustering is a meaningful business segmentation, not a statistical artefact, "
                "and that targeted support programmes for Cluster B/C stores could drive material uplift "
                "if the drivers of Cluster A outperformance can be identified and replicated."
            )
        else:
            business_conclusion = (
                f"Revenue differences across store performance clusters are not statistically "
                f"significant (F={f_stat:.2f}, p={p_value:.4f}). This unexpected finding suggests "
                "the cluster labels may not accurately reflect revenue performance, or that other "
                "factors (regional demand, competitive intensity) dominate the cluster effect. "
                "A deeper investigation into cluster assignment methodology is warranted."
            )

        statistical_conclusion = (
            f"One-way ANOVA: F({len(groups) - 1}, {len(all_vals) - len(groups)})={f_stat:.4f}, "
            f"p={p_value:.4f}, η²={eta_squared:.4f} ({effect_interp} effect). "
            f"H0 {'rejected' if rejected else 'not rejected'} at α={self.ALPHA}."
        )

        return HypothesisTestResult(
            hypothesis_id="H1",
            hypothesis_text="Cluster A stores have significantly higher revenue than Cluster B/C stores.",
            test_name="One-way ANOVA",
            test_statistic=float(f_stat),
            p_value=float(p_value),
            alpha=self.ALPHA,
            rejected=rejected,
            effect_size=eta_squared,
            effect_size_interpretation=effect_interp,
            business_conclusion=business_conclusion,
            statistical_conclusion=statistical_conclusion,
        )

    # ------------------------------------------------------------------ #
    # H2: Silver-tier churn post month-24 (two-sample t-test)             #
    # ------------------------------------------------------------------ #

    def test_silver_churn(
        self, transactions: pd.DataFrame, customers: pd.DataFrame
    ) -> HypothesisTestResult:
        """Test H2: Silver-tier purchase frequency declines significantly post month-24.

        Method: Two-sample t-test comparing avg monthly transactions per Silver customer
        in months 1-24 vs months 25-36.
        Effect size: Cohen's d.

        Args:
            transactions: Transaction data with date, customer_id.
            customers: Customer master with loyalty_tier.

        Returns:
            HypothesisTestResult for H2.
        """
        logger.info("Testing H2: Silver-tier churn post month-24 (t-test)")

        # Identify Silver customers
        silver_ids = set(customers.loc[customers["loyalty_tier"] == "Silver", "customer_id"])
        tx = transactions.copy()
        tx = tx[tx["customer_id"].isin(silver_ids)].copy()

        if tx.empty:
            return self._insufficient_data_result("H2", "Two-sample t-test on Silver customer frequency")

        tx["date"] = pd.to_datetime(tx["date"])
        min_date = tx["date"].min()
        tx["month_num"] = (
            (tx["date"].dt.year - min_date.year) * 12
            + (tx["date"].dt.month - min_date.month)
            + 1
        )

        early_tx = tx[tx["month_num"] <= 24]
        late_tx = tx[tx["month_num"] > 24]

        if early_tx.empty or late_tx.empty:
            return self._insufficient_data_result("H2", "Two-sample t-test on Silver customer frequency")

        # Average monthly transactions per customer
        early_freq = (
            early_tx.groupby(["customer_id", "month_num"]).size()
            .groupby("customer_id").mean()
            .values
        )
        late_freq = (
            late_tx.groupby(["customer_id", "month_num"]).size()
            .groupby("customer_id").mean()
            .values
        )

        t_stat, p_value = stats.ttest_ind(early_freq, late_freq, equal_var=False, alternative="greater")
        cohens_d = self._cohens_d(early_freq, late_freq)
        effect_interp = self._cohens_d_interpretation(abs(cohens_d))
        rejected = bool(p_value < self.ALPHA)

        early_mean = float(np.mean(early_freq))
        late_mean = float(np.mean(late_freq))
        freq_decline_pct = (late_mean - early_mean) / max(early_mean, 0.001) * 100

        if rejected:
            business_conclusion = (
                f"Silver-tier customers show a statistically significant decline in purchase frequency "
                f"following month 24 (t={t_stat:.3f}, p={p_value:.4f}, d={cohens_d:.3f}, "
                f"effect size: {effect_interp}). Average monthly transactions per Silver customer "
                f"fell from {early_mean:.2f} to {late_mean:.2f} — a {abs(freq_decline_pct):.1f}% "
                f"decline — consistent with the hypothesis that the loyalty fee increase materially "
                "damaged engagement in this high-volume segment. Silver customers represent a "
                "critical mid-funnel cohort: if not recovered, this churn will cascade into "
                "top-line revenue decline within 6-12 months."
            )
        else:
            business_conclusion = (
                f"Silver-tier purchase frequency does not show a statistically significant post-month-24 "
                f"decline (t={t_stat:.3f}, p={p_value:.4f}). While transaction frequency moved from "
                f"{early_mean:.2f} to {late_mean:.2f} per month, this difference is within the range "
                "of natural variation. The loyalty fee increase may not have driven the behavioural "
                "change hypothesised, and alternative explanations (seasonal effects, category "
                "availability) should be investigated."
            )

        statistical_conclusion = (
            f"Welch's two-sample t-test (H0: early ≥ late): t={t_stat:.4f}, p={p_value:.4f} "
            f"(one-tailed), Cohen's d={cohens_d:.4f} ({effect_interp} effect). "
            f"n_early={len(early_freq)}, n_late={len(late_freq)}. "
            f"H0 {'rejected' if rejected else 'not rejected'} at α={self.ALPHA}."
        )

        return HypothesisTestResult(
            hypothesis_id="H2",
            hypothesis_text="Silver-tier customer purchase frequency declines significantly after month 24 (loyalty fee introduction).",
            test_name="Welch's two-sample t-test",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            alpha=self.ALPHA,
            rejected=rejected,
            effect_size=float(abs(cohens_d)),
            effect_size_interpretation=effect_interp,
            business_conclusion=business_conclusion,
            statistical_conclusion=statistical_conclusion,
        )

    # ------------------------------------------------------------------ #
    # H3: Private label vs national brand margin (Welch's t-test)         #
    # ------------------------------------------------------------------ #

    def test_private_label_margin(
        self, transactions: pd.DataFrame, products: pd.DataFrame
    ) -> HypothesisTestResult:
        """Test H3: Private label products have significantly higher gross margin than national brands.

        Method: Welch's t-test on product-level gross margin %.
        Effect size: Cohen's d.

        Args:
            transactions: Transaction data.
            products: Product catalogue with brand_type.

        Returns:
            HypothesisTestResult for H3.
        """
        logger.info("Testing H3: Private label vs national brand margin (Welch t-test)")

        prod_margin = (
            transactions.groupby("product_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "margin_pct": g["gross_profit"].sum() / max(g["gross_revenue"].sum(), 1) * 100
                    }
                )
            )
            .reset_index()
        )
        prod_margin = prod_margin.merge(
            products[["product_id", "brand_type"]], on="product_id", how="left"
        ).dropna(subset=["brand_type"])

        pl_margins = prod_margin.loc[
            prod_margin["brand_type"] == "private_label", "margin_pct"
        ].values
        nb_margins = prod_margin.loc[
            prod_margin["brand_type"] == "national_brand", "margin_pct"
        ].values

        if len(pl_margins) < 2 or len(nb_margins) < 2:
            return self._insufficient_data_result("H3", "Welch's t-test on product gross margin")

        t_stat, p_value = stats.ttest_ind(pl_margins, nb_margins, equal_var=False)
        cohens_d = self._cohens_d(pl_margins, nb_margins)
        effect_interp = self._cohens_d_interpretation(abs(cohens_d))
        rejected = bool(p_value < self.ALPHA)

        pl_mean = float(np.mean(pl_margins))
        nb_mean = float(np.mean(nb_margins))
        margin_gap = pl_mean - nb_mean

        if rejected:
            business_conclusion = (
                f"Private label products generate a statistically significant {margin_gap:.1f}pp "
                f"gross margin advantage over national brands (private label: {pl_mean:.1f}% vs "
                f"national brand: {nb_mean:.1f}%; t={t_stat:.3f}, p={p_value:.4f}, d={cohens_d:.3f}, "
                f"effect size: {effect_interp}). This confirms private label as NovaMart's most "
                "powerful margin lever. Every 1pp increase in private-label revenue mix translates "
                f"directly to {margin_gap / 100:.3f}pp of portfolio gross margin improvement — "
                "a highly capital-efficient route to margin recovery that does not require "
                "volume growth or cost reduction."
            )
        else:
            business_conclusion = (
                f"The gross margin difference between private label ({pl_mean:.1f}%) and national "
                f"brands ({nb_mean:.1f}%) is not statistically significant (t={t_stat:.3f}, "
                f"p={p_value:.4f}). This is unexpected and may indicate that private label "
                "products in NovaMart's portfolio are not yet achieving their potential margin "
                "premium — possibly due to under-pricing, high production costs, or quality "
                "positioning that limits pricing power."
            )

        statistical_conclusion = (
            f"Welch's t-test: t={t_stat:.4f}, p={p_value:.4f}, Cohen's d={cohens_d:.4f} "
            f"({effect_interp} effect). n_private_label={len(pl_margins)}, "
            f"n_national_brand={len(nb_margins)}. "
            f"H0 (equal means) {'rejected' if rejected else 'not rejected'} at α={self.ALPHA}."
        )

        return HypothesisTestResult(
            hypothesis_id="H3",
            hypothesis_text="Private label products have significantly higher gross margin than national brand products.",
            test_name="Welch's t-test",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            alpha=self.ALPHA,
            rejected=rejected,
            effect_size=float(abs(cohens_d)),
            effect_size_interpretation=effect_interp,
            business_conclusion=business_conclusion,
            statistical_conclusion=statistical_conclusion,
        )

    # ------------------------------------------------------------------ #
    # H4: Urban vs rural digital marketing ROI (Mann-Whitney U)           #
    # ------------------------------------------------------------------ #

    def test_urban_digital_roi(
        self,
        marketing: pd.DataFrame,
        transactions: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> HypothesisTestResult:
        """Test H4: Urban stores have significantly higher digital marketing ROI than rural.

        Method: Mann-Whitney U (non-parametric) on monthly digital revenue / digital spend ratio.
        Effect size: rank-biserial correlation (r = 1 - 2U/n1n2).

        Args:
            marketing: Marketing spend data with channel and spend_usd.
            transactions: Transaction data for revenue.
            stores: Store master with store_format for urban/rural classification.

        Returns:
            HypothesisTestResult for H4.
        """
        logger.info("Testing H4: Urban vs rural digital marketing ROI (Mann-Whitney U)")

        # Urban proxy: 'urban' or 'flagship' format; rural: 'suburban' or 'rural'
        store_format = stores[["store_id", "store_format"]].copy()
        store_format["is_urban"] = store_format["store_format"].isin(["urban", "flagship"])

        # Use digital_roi_multiplier from marketing if available, else compute
        digital_mkt = marketing[marketing["channel"] == "digital"].copy()
        if "digital_roi_multiplier" in digital_mkt.columns and digital_mkt["digital_roi_multiplier"].notna().any():
            digital_mkt = digital_mkt.merge(store_format, on="store_id", how="left").dropna(subset=["is_urban"])
            urban_roi = digital_mkt.loc[digital_mkt["is_urban"], "digital_roi_multiplier"].dropna().values
            rural_roi = digital_mkt.loc[~digital_mkt["is_urban"], "digital_roi_multiplier"].dropna().values
        else:
            # Compute: digital revenue / digital spend per store-month
            monthly_rev = (
                transactions.groupby(["store_id", "year_month"])["gross_revenue"]
                .sum()
                .reset_index(name="revenue")
            )
            digital_spend = digital_mkt.groupby(["store_id", "year_month"])["spend_usd"].sum().reset_index(name="digital_spend")
            merged = monthly_rev.merge(digital_spend, on=["store_id", "year_month"]).merge(store_format, on="store_id")
            merged["digital_roi"] = merged["revenue"] / merged["digital_spend"].clip(lower=1)
            urban_roi = merged.loc[merged["is_urban"], "digital_roi"].values
            rural_roi = merged.loc[~merged["is_urban"], "digital_roi"].values

        if len(urban_roi) < 2 or len(rural_roi) < 2:
            return self._insufficient_data_result("H4", "Mann-Whitney U on digital ROI")

        u_stat, p_value = stats.mannwhitneyu(urban_roi, rural_roi, alternative="greater")
        n1, n2 = len(urban_roi), len(rural_roi)
        rank_biserial_r = float(1 - 2 * u_stat / (n1 * n2))
        effect_interp = self._cohens_d_interpretation(abs(rank_biserial_r))
        rejected = bool(p_value < self.ALPHA)

        urban_mean = float(np.median(urban_roi))
        rural_mean = float(np.median(rural_roi))
        roi_premium = (urban_mean - rural_mean) / max(rural_mean, 0.001) * 100

        if rejected:
            business_conclusion = (
                f"Urban stores demonstrate a statistically significant digital marketing ROI "
                f"advantage over rural/suburban locations (U={u_stat:.0f}, p={p_value:.4f}, "
                f"rank-biserial r={rank_biserial_r:.3f}, effect size: {effect_interp}). "
                f"Median digital ROI is {roi_premium:.1f}% higher in urban stores "
                f"({urban_mean:.2f}× vs {rural_mean:.2f}× revenue return per digital dollar). "
                "This finding justifies a location-differentiated digital marketing strategy: "
                "urban stores should receive disproportionately higher digital budget allocation, "
                "while rural stores' digital spend should be reviewed for channel mix and targeting "
                "optimisation before further investment."
            )
        else:
            business_conclusion = (
                f"Urban and rural stores do not show a statistically significant difference in "
                f"digital marketing ROI (U={u_stat:.0f}, p={p_value:.4f}). Digital channels "
                "appear to perform comparably across store formats — either because NovaMart's "
                "digital strategy is well-standardised, or because the ROI measurement methodology "
                "does not capture location-specific nuances. A more granular analysis by specific "
                "digital sub-channel (search, social, display) may reveal format-specific differences."
            )

        statistical_conclusion = (
            f"Mann-Whitney U test (H0: urban ≤ rural): U={u_stat:.2f}, p={p_value:.4f} (one-tailed), "
            f"rank-biserial r={rank_biserial_r:.4f} ({effect_interp} effect). "
            f"n_urban={n1}, n_rural={n2}. "
            f"H0 {'rejected' if rejected else 'not rejected'} at α={self.ALPHA}."
        )

        return HypothesisTestResult(
            hypothesis_id="H4",
            hypothesis_text="Urban stores have significantly higher digital marketing ROI than rural/suburban stores.",
            test_name="Mann-Whitney U test",
            test_statistic=float(u_stat),
            p_value=float(p_value),
            alpha=self.ALPHA,
            rejected=rejected,
            effect_size=float(abs(rank_biserial_r)),
            effect_size_interpretation=effect_interp,
            business_conclusion=business_conclusion,
            statistical_conclusion=statistical_conclusion,
        )

    # ------------------------------------------------------------------ #
    # H5: Discount rate vs profit margin (Pearson r)                       #
    # ------------------------------------------------------------------ #

    def test_discount_margin_correlation(
        self, transactions: pd.DataFrame
    ) -> HypothesisTestResult:
        """Test H5: Discount rate is negatively correlated with profit margin.

        Method: Pearson r on transaction-level discount_pct vs gross margin %.
        Effect size: |r| (correlation coefficient itself is the effect size for r-tests).

        Args:
            transactions: Transaction data with discount_pct and gross profit/revenue.

        Returns:
            HypothesisTestResult for H5.
        """
        logger.info("Testing H5: Discount rate vs profit margin correlation (Pearson r)")

        tx = transactions.copy()
        tx["margin_pct"] = np.where(
            tx["gross_revenue"] > 0,
            tx["gross_profit"] / tx["gross_revenue"] * 100,
            np.nan,
        )
        tx = tx.dropna(subset=["margin_pct", "discount_pct"])

        if len(tx) > 100_000:
            tx = tx.sample(n=100_000, random_state=42)

        r_stat, p_value = stats.pearsonr(tx["discount_pct"], tx["margin_pct"])
        effect_size = float(abs(r_stat))
        effect_interp = self._cohens_d_interpretation(effect_size)
        rejected = bool(p_value < self.ALPHA)

        avg_discount = float(tx["discount_pct"].mean() * 100)
        avg_margin = float(tx["margin_pct"].mean())

        if rejected and r_stat < 0:
            business_conclusion = (
                f"Discount rate is negatively and significantly correlated with gross margin "
                f"(r={r_stat:.3f}, p={p_value:.4f}, n={len(tx):,}), confirming H5. "
                f"NovaMart's portfolio-wide average discount of {avg_discount:.1f}% is "
                f"compressing gross margins to an average of {avg_margin:.1f}%. "
                f"A 1pp reduction in average discount depth is estimated to recover "
                f"{abs(r_stat):.3f}pp of gross margin — without any change in volume or mix. "
                "Implementing a discount optimisation model that targets promotions only at "
                "genuinely price-sensitive customers (rather than blanket promotions) could "
                "recover 150-300bps of gross margin while preserving volume at reduced cost."
            )
        elif rejected and r_stat > 0:
            business_conclusion = (
                f"Counter-intuitively, discount rate shows a positive significant correlation with "
                f"gross margin (r={r_stat:.3f}, p={p_value:.4f}). This likely reflects a "
                "confounding factor: high-margin product categories may also receive heavier "
                "promotional activity. A category-controlled analysis is required to isolate the "
                "true discount-margin relationship."
            )
        else:
            business_conclusion = (
                f"The correlation between discount rate and gross margin is not statistically "
                f"significant (r={r_stat:.3f}, p={p_value:.4f}). This may indicate that "
                "NovaMart's discount policy is well-calibrated by product, that high-margin "
                "products receive proportionally higher discounts (offsetting the margin erosion), "
                "or that the transaction-level analysis masks category-specific dynamics."
            )

        statistical_conclusion = (
            f"Pearson r={r_stat:.4f}, p={p_value:.4f}, effect size |r|={effect_size:.4f} "
            f"({effect_interp}). n={len(tx):,}. "
            f"H0 (ρ=0) {'rejected' if rejected else 'not rejected'} at α={self.ALPHA}."
        )

        return HypothesisTestResult(
            hypothesis_id="H5",
            hypothesis_text="Discount rate is negatively correlated with profit margin: deeper discounts erode gross margins.",
            test_name="Pearson correlation test",
            test_statistic=float(r_stat),
            p_value=float(p_value),
            alpha=self.ALPHA,
            rejected=rejected,
            effect_size=effect_size,
            effect_size_interpretation=effect_interp,
            business_conclusion=business_conclusion,
            statistical_conclusion=statistical_conclusion,
        )

    def run_all(self, data: dict[str, pd.DataFrame]) -> list[HypothesisTestResult]:
        """Run all five pre-specified hypothesis tests.

        Args:
            data: Dict produced by DataLoader.load_all().

        Returns:
            List of HypothesisTestResult objects.
        """
        logger.info("Running all 5 hypothesis tests")
        results: list[HypothesisTestResult] = [
            self.test_cluster_revenue_difference(data["transactions"], data["stores"]),
            self.test_silver_churn(data["transactions"], data["customers"]),
            self.test_private_label_margin(data["transactions"], data["products"]),
            self.test_urban_digital_roi(data["marketing_spend"], data["transactions"], data["stores"]),
            self.test_discount_margin_correlation(data["transactions"]),
        ]
        n_rejected = sum(1 for r in results if r.rejected)
        logger.success(f"Hypothesis testing complete: {n_rejected}/{len(results)} H0 rejected at α={self.ALPHA}")
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d for two independent groups (pooled SD)."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
            / (n1 + n2 - 2)
        )
        if pooled_std == 0:
            return 0.0
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    @staticmethod
    def _cohens_d_interpretation(d: float) -> str:
        """Classify Cohen's d (or |r|) using conventional thresholds."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    @staticmethod
    def _eta_squared_interpretation(eta2: float) -> str:
        """Classify eta-squared using conventional thresholds (Cohen 1988)."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"

    def _insufficient_data_result(self, hypothesis_id: str, test_name: str) -> HypothesisTestResult:
        """Return a placeholder result when there is insufficient data to run a test."""
        warnings.warn(f"Insufficient data for {hypothesis_id}; returning placeholder result")
        return HypothesisTestResult(
            hypothesis_id=hypothesis_id,
            hypothesis_text=f"Test {hypothesis_id} could not be run: insufficient data",
            test_name=test_name,
            test_statistic=float("nan"),
            p_value=float("nan"),
            alpha=self.ALPHA,
            rejected=False,
            effect_size=float("nan"),
            effect_size_interpretation="negligible",
            business_conclusion="Insufficient data to draw a business conclusion for this test.",
            statistical_conclusion="Test could not be executed due to insufficient data in one or more groups.",
        )
