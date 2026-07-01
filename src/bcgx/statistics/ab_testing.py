"""A/B test analysis framework for NovaMart natural experiments.

Primary use case: analyse the loyalty fee increase (month 25+) as a natural experiment,
treating Silver-tier customers as the treatment group and Gold-tier customers as the
control group — on the assumption that Gold-tier customers were not subject to the
same fee change and therefore serve as a credible counterfactual.

Statistical methods:
- Two-sample t-test (scipy.stats.ttest_ind) for group mean comparison
- 95% confidence interval on the lift using t-distribution
- Post-hoc power analysis (statsmodels) to assess whether the test was adequately powered
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class ABTestResult:
    """Result of an A/B test or natural experiment analysis."""

    test_name: str
    treatment_description: str
    control_description: str
    metric: str
    treatment_mean: float
    control_mean: float
    absolute_lift: float  # treatment_mean - control_mean
    relative_lift_pct: float  # (treatment_mean - control_mean) / control_mean * 100
    p_value: float
    confidence_interval: tuple[float, float]  # 95% CI on absolute lift
    statistical_power: float  # observed power (post-hoc)
    sample_size_treatment: int
    sample_size_control: int
    business_impact: str  # dollar / business narrative of observed effect


class ABTestAnalyzer:
    """Analyse natural experiments and A/B tests for NovaMart interventions.

    Usage::

        from bcgx.data.loader import DataLoader
        from bcgx.statistics.ab_testing import ABTestAnalyzer

        data = DataLoader("data/raw").load_all()
        analyzer = ABTestAnalyzer()
        results = analyzer.run_all(data)
    """

    ALPHA: float = 0.05
    POWER_TARGET: float = 0.80

    # ------------------------------------------------------------------ #
    # Primary analysis: loyalty fee natural experiment                    #
    # ------------------------------------------------------------------ #

    def analyze_loyalty_fee_impact(
        self, transactions: pd.DataFrame, customers: pd.DataFrame
    ) -> ABTestResult:
        """Analyse the loyalty fee increase as a natural experiment.

        Design:
        - Treatment: Silver-tier customers in months 25-36 of data
        - Control: Gold-tier customers in months 25-36 of data
        - Metric: Average monthly transactions per customer
        - Assumption: Gold-tier customers were not subject to the same fee structure
          change and therefore provide a valid counterfactual for purchase frequency
          in the absence of the intervention.

        Limitations: Parallel trends assumption is not formally tested here (would
        require difference-in-differences). Interpret results with appropriate caution.

        Args:
            transactions: Transaction data with date, customer_id.
            customers: Customer master with loyalty_tier.

        Returns:
            ABTestResult for the loyalty fee natural experiment.
        """
        logger.info("A/B analysis: loyalty fee impact (Silver vs Gold, months 25-36)")

        tx = transactions.copy()
        tx["date"] = pd.to_datetime(tx["date"])

        min_date = tx["date"].min()
        tx["month_num"] = (
            (tx["date"].dt.year - min_date.year) * 12
            + (tx["date"].dt.month - min_date.month)
            + 1
        )

        # Post-intervention window
        post_tx = tx[tx["month_num"] > 24].copy()

        if post_tx.empty:
            logger.warning("No transactions found in months 25+; using full dataset")
            post_tx = tx.copy()

        # Tag loyalty tier
        tier_map = customers.set_index("customer_id")["loyalty_tier"]
        post_tx["loyalty_tier"] = post_tx["customer_id"].map(tier_map)

        silver_tx = post_tx[post_tx["loyalty_tier"] == "Silver"]
        gold_tx = post_tx[post_tx["loyalty_tier"] == "Gold"]

        if silver_tx.empty or gold_tx.empty:
            logger.warning("Missing Silver or Gold transactions in post-intervention window")
            # Return a placeholder result
            return self._empty_result(
                test_name="Loyalty Fee Natural Experiment",
                reason="Insufficient post-intervention data for Silver or Gold tier",
            )

        # Average monthly transactions per customer
        silver_monthly = (
            silver_tx.groupby(["customer_id", "month_num"]).size()
            .groupby("customer_id").mean()
            .values
        )
        gold_monthly = (
            gold_tx.groupby(["customer_id", "month_num"]).size()
            .groupby("customer_id").mean()
            .values
        )

        t_stat, p_value = stats.ttest_ind(silver_monthly, gold_monthly, equal_var=False)

        treatment_mean = float(np.mean(silver_monthly))
        control_mean = float(np.mean(gold_monthly))
        absolute_lift = treatment_mean - control_mean
        relative_lift_pct = (absolute_lift / max(abs(control_mean), 1e-9)) * 100

        # 95% CI on absolute lift via Welch's interval
        ci = self._welch_ci(silver_monthly, gold_monthly, alpha=self.ALPHA)

        # Post-hoc power (observed)
        pooled_std = float(
            np.sqrt(
                (np.var(silver_monthly, ddof=1) + np.var(gold_monthly, ddof=1)) / 2
            )
        )
        power = self._compute_power(
            n1=len(silver_monthly),
            n2=len(gold_monthly),
            effect_size=abs(absolute_lift) / max(pooled_std, 1e-9),
            alpha=self.ALPHA,
        )

        # Business impact narrative
        n_silver_customers = len(silver_monthly)
        total_txns_lost = abs(absolute_lift) * n_silver_customers
        # Estimate revenue per transaction from data
        avg_txn_value = float(silver_tx["gross_revenue"].sum() / max(len(silver_tx), 1))
        estimated_monthly_revenue_impact = total_txns_lost * avg_txn_value
        is_sig = bool(p_value < self.ALPHA)

        if is_sig and absolute_lift < 0:
            business_impact = (
                f"Silver-tier customers in the post-intervention period average "
                f"{treatment_mean:.2f} transactions/month versus {control_mean:.2f} for Gold-tier "
                f"customers — a {abs(relative_lift_pct):.1f}% gap (p={p_value:.4f}). "
                f"Across {n_silver_customers:,} Silver customers, this frequency decline represents "
                f"approximately {total_txns_lost:,.0f} lost transactions per month, equivalent to "
                f"~${estimated_monthly_revenue_impact:,.0f} in monthly revenue at-risk. "
                "If this gap persists over a full year and Silver customers progressively disengage, "
                f"the cumulative annual revenue impact is estimated at "
                f"~${estimated_monthly_revenue_impact * 12 / 1e6:.1f}M — a materially significant "
                "outcome of the loyalty fee introduction that warrants immediate management response."
            )
        elif not is_sig:
            business_impact = (
                f"No statistically significant difference in purchase frequency between Silver "
                f"({treatment_mean:.2f} tx/month) and Gold ({control_mean:.2f} tx/month) customers "
                f"in the post-intervention period (p={p_value:.4f}). The loyalty fee change does not "
                "appear to have materially impacted Silver-tier purchase frequency relative to the "
                "Gold-tier counterfactual, though this conclusion carries the caveat that the "
                "parallel trends assumption is not validated here."
            )
        else:
            business_impact = (
                f"Silver-tier customers show {abs(relative_lift_pct):.1f}% {'higher' if absolute_lift > 0 else 'lower'} "
                f"purchase frequency than Gold-tier post-intervention (p={p_value:.4f}). "
                "This directional signal warrants deeper investigation with a more rigorous "
                "difference-in-differences design to control for pre-existing tier differences."
            )

        return ABTestResult(
            test_name="Loyalty Fee Natural Experiment (Silver vs Gold, Months 25-36)",
            treatment_description="Silver-tier customers in months 25-36 (exposed to loyalty fee increase)",
            control_description="Gold-tier customers in months 25-36 (not subject to same fee structure change)",
            metric="avg_monthly_transactions_per_customer",
            treatment_mean=round(treatment_mean, 4),
            control_mean=round(control_mean, 4),
            absolute_lift=round(absolute_lift, 4),
            relative_lift_pct=round(relative_lift_pct, 2),
            p_value=round(float(p_value), 6),
            confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
            statistical_power=round(power, 4),
            sample_size_treatment=len(silver_monthly),
            sample_size_control=len(gold_monthly),
            business_impact=business_impact,
        )

    def compute_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Compute the minimum sample size per group for a two-sample t-test.

        Uses the standard normal approximation for proportions or rates.

        Args:
            baseline_rate: Expected value in the control group (e.g. 3.2 transactions/month).
            minimum_detectable_effect: Smallest practically meaningful absolute difference
                (e.g. 0.5 means we want to detect a 0.5 tx/month difference).
            alpha: Type I error rate (default 0.05).
            power: Desired statistical power (default 0.80).

        Returns:
            Required sample size per group (integer).
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Assume standard deviation ≈ sqrt(baseline_rate) for count data (Poisson approximation)
        sigma = math.sqrt(max(baseline_rate, 1e-9))
        effect_size = minimum_detectable_effect / max(sigma, 1e-9)

        n = ((z_alpha + z_beta) / effect_size) ** 2
        return int(math.ceil(n))

    def run_all(self, data: dict[str, pd.DataFrame]) -> list[ABTestResult]:
        """Run all A/B test analyses.

        Args:
            data: Dict produced by DataLoader.load_all().

        Returns:
            List of ABTestResult objects.
        """
        logger.info("Running all A/B test analyses")
        results: list[ABTestResult] = [
            self.analyze_loyalty_fee_impact(data["transactions"], data["customers"]),
        ]
        logger.success(f"A/B testing complete: {len(results)} analyses run")
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _welch_ci(
        group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Compute Welch's 95% confidence interval on the difference of means (g1 - g2)."""
        m1, m2 = np.mean(group1), np.mean(group2)
        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        se = math.sqrt(s1 / n1 + s2 / n2)
        df = (s1 / n1 + s2 / n2) ** 2 / (
            (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        )
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))

        diff = float(m1 - m2)
        return (diff - t_crit * se, diff + t_crit * se)

    @staticmethod
    def _compute_power(
        n1: int, n2: int, effect_size: float, alpha: float = 0.05
    ) -> float:
        """Estimate post-hoc statistical power using the normal approximation.

        Args:
            n1: Sample size of group 1.
            n2: Sample size of group 2.
            effect_size: Cohen's d (observed).
            alpha: Type I error rate.

        Returns:
            Estimated power between 0 and 1.
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        # Harmonic mean sample size
        n_harmonic = 2 * n1 * n2 / max(n1 + n2, 1)
        ncp = effect_size * math.sqrt(n_harmonic / 2)
        power = float(1 - stats.norm.cdf(z_alpha - ncp))
        return min(max(power, 0.0), 1.0)

    @staticmethod
    def _empty_result(test_name: str, reason: str) -> ABTestResult:
        """Return a placeholder ABTestResult when analysis cannot be performed."""
        return ABTestResult(
            test_name=test_name,
            treatment_description="N/A",
            control_description="N/A",
            metric="avg_monthly_transactions_per_customer",
            treatment_mean=float("nan"),
            control_mean=float("nan"),
            absolute_lift=float("nan"),
            relative_lift_pct=float("nan"),
            p_value=float("nan"),
            confidence_interval=(float("nan"), float("nan")),
            statistical_power=float("nan"),
            sample_size_treatment=0,
            sample_size_control=0,
            business_impact=f"Analysis could not be completed: {reason}",
        )
