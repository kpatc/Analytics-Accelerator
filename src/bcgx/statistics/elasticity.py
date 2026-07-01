"""Price elasticity estimation for NovaMart product categories.

Method: log-log OLS regression — regress log(quantity) on log(effective_price)
using discount variation as the price instrument (since observed price variation
is largely driven by promotional discounting, which is under NovaMart's control).

Elasticity interpretation:
- |e| > 1: elastic — customers are price-sensitive; a price increase reduces total revenue
- |e| < 1: inelastic — customers are price-insensitive; a price increase raises total revenue
- |e| ≈ 1: unit elastic — price changes leave total revenue unchanged

Uses statsmodels OLS to provide proper confidence intervals, p-values, and R².
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger


@dataclass
class ElasticityEstimate:
    """Price elasticity estimate for a single product category."""

    category: str
    elasticity_coefficient: float  # typically negative (e.g. -1.8)
    elasticity_std_error: float
    confidence_interval_lower: float  # 95% CI lower bound
    confidence_interval_upper: float  # 95% CI upper bound
    r_squared: float
    n_observations: int
    interpretation: str  # "elastic" | "inelastic" | "unit elastic"
    pricing_recommendation: str
    revenue_impact_of_10pct_increase: float  # % revenue change if price +10%


class ElasticityAnalyzer:
    """Estimate price elasticity by product category using log-log OLS regression.

    Uses discount_pct variation as the price instrument — since NovaMart controls
    discount depth (not shelf price), discount variation is the cleaner source of
    price variation in the data.

    Effective price = unit_price × (1 - discount_pct)
    """

    MIN_OBS: int = 30  # minimum observations per category for reliable estimation
    CI_ALPHA: float = 0.05  # 95% confidence intervals

    def estimate_category_elasticity(
        self, transactions: pd.DataFrame, products: pd.DataFrame
    ) -> list[ElasticityEstimate]:
        """Estimate price elasticity for each product category.

        For each category, fits: log(quantity) ~ log(effective_price) using OLS.
        Categories with fewer than MIN_OBS observations are skipped.

        Args:
            transactions: Transaction data with quantity, unit_price, discount_pct, product_id.
            products: Product catalogue with product_id, category.

        Returns:
            List of ElasticityEstimate, one per estimable category.
        """
        logger.info("Estimating price elasticity by product category (log-log OLS)")

        tx = transactions.copy()
        tx = tx.merge(products[["product_id", "category"]], on="product_id", how="left")

        # Compute effective (net) price
        tx["effective_price"] = tx["unit_price"] * (1 - tx["discount_pct"].clip(lower=0, upper=0.99))

        # Drop zero or negative prices / quantities
        tx = tx[(tx["effective_price"] > 0) & (tx["quantity"] > 0)].copy()

        tx["log_quantity"] = np.log(tx["quantity"].astype(float))
        tx["log_price"] = np.log(tx["effective_price"])

        results: list[ElasticityEstimate] = []
        categories = tx["category"].dropna().unique()
        logger.info(f"Estimating elasticity for {len(categories)} categories")

        for cat in sorted(categories):
            cat_data = tx[tx["category"] == cat].dropna(subset=["log_quantity", "log_price"])

            if len(cat_data) < self.MIN_OBS:
                logger.debug(f"Skipping category '{cat}': only {len(cat_data)} observations (min={self.MIN_OBS})")
                continue

            # Check for price variation — no variation → can't estimate elasticity
            if cat_data["log_price"].std() < 1e-6:
                logger.debug(f"Skipping category '{cat}': insufficient price variation")
                continue

            try:
                estimate = self._fit_ols(cat, cat_data["log_price"].values, cat_data["log_quantity"].values)
                if estimate is not None:
                    results.append(estimate)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"OLS failed for category '{cat}': {exc}")

        logger.info(f"Elasticity estimation complete: {len(results)} categories estimated")
        return results

    def run_all(self, data: dict[str, pd.DataFrame]) -> list[ElasticityEstimate]:
        """Estimate elasticity for all categories.

        Args:
            data: Dict produced by DataLoader.load_all().

        Returns:
            List of ElasticityEstimate objects.
        """
        return self.estimate_category_elasticity(data["transactions"], data["products"])

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fit_ols(
        self, category: str, log_price: np.ndarray, log_quantity: np.ndarray
    ) -> ElasticityEstimate | None:
        """Fit log-log OLS and extract elasticity estimate with confidence intervals.

        Args:
            category: Category name (for labelling).
            log_price: Log-transformed effective prices.
            log_quantity: Log-transformed quantities.

        Returns:
            ElasticityEstimate or None if the regression fails.
        """
        X = sm.add_constant(log_price)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(log_quantity, X).fit()

        # Support both older (array) and newer (Series) statsmodels param formats
        params = model.params
        bse = model.bse
        if hasattr(params, "iloc"):
            elasticity = float(params.iloc[1])
            std_error = float(bse.iloc[1])
        else:
            elasticity = float(params[1])
            std_error = float(bse[1])
        ci = model.conf_int(alpha=self.CI_ALPHA)
        if hasattr(ci, "iloc"):
            ci_lower = float(ci.iloc[1, 0])
            ci_upper = float(ci.iloc[1, 1])
        else:
            ci_lower = float(ci[1, 0])
            ci_upper = float(ci[1, 1])
        r_squared = float(model.rsquared)
        n_obs = int(model.nobs)

        interpretation = self._interpret_elasticity(elasticity)
        recommendation = self._pricing_recommendation(category, elasticity, interpretation)
        revenue_impact = self._revenue_impact_10pct_increase(elasticity)

        return ElasticityEstimate(
            category=category,
            elasticity_coefficient=round(elasticity, 4),
            elasticity_std_error=round(std_error, 4),
            confidence_interval_lower=round(ci_lower, 4),
            confidence_interval_upper=round(ci_upper, 4),
            r_squared=round(r_squared, 4),
            n_observations=n_obs,
            interpretation=interpretation,
            pricing_recommendation=recommendation,
            revenue_impact_of_10pct_increase=round(revenue_impact, 2),
        )

    @staticmethod
    def _interpret_elasticity(elasticity: float) -> str:
        """Classify elasticity magnitude."""
        abs_e = abs(elasticity)
        if abs_e > 1.0:
            return "elastic"
        elif abs_e < 1.0:
            return "inelastic"
        else:
            return "unit elastic"

    @staticmethod
    def _revenue_impact_10pct_increase(elasticity: float) -> float:
        """Estimate % revenue change from a 10% price increase.

        Revenue change % = (1 + elasticity × 0.10) × 1.10 - 1.0, expressed as %.
        For elastic goods (|e|>1), revenue falls; for inelastic, revenue rises.

        Args:
            elasticity: Price elasticity coefficient (typically negative).

        Returns:
            Percentage change in revenue (negative = revenue loss).
        """
        quantity_change_pct = elasticity * 0.10  # % change in quantity
        price_change_pct = 0.10  # 10% price increase
        revenue_change_pct = (1 + price_change_pct) * (1 + quantity_change_pct) - 1.0
        return round(revenue_change_pct * 100, 2)

    @staticmethod
    def _pricing_recommendation(category: str, elasticity: float, interpretation: str) -> str:
        """Generate a category-specific pricing recommendation.

        Args:
            category: Product category name.
            elasticity: Estimated elasticity coefficient.
            interpretation: "elastic" | "inelastic" | "unit elastic".

        Returns:
            One-sentence pricing recommendation for NovaMart management.
        """
        abs_e = abs(elasticity)
        revenue_impact = ElasticityAnalyzer._revenue_impact_10pct_increase(elasticity)

        if interpretation == "elastic":
            return (
                f"{category} demand is price-elastic (ε={elasticity:.2f}): a 10% price increase "
                f"would reduce revenue by approximately {abs(revenue_impact):.1f}%. "
                "Avoid broad-based price increases; instead focus on targeted promotions for "
                "price-sensitive customer segments while maintaining full price for low-sensitivity cohorts."
            )
        elif interpretation == "inelastic":
            return (
                f"{category} demand is price-inelastic (ε={elasticity:.2f}): a 10% price increase "
                f"would increase revenue by approximately {revenue_impact:.1f}%. "
                "This category has significant untapped pricing power — test a 3-5% price increase "
                "in 20% of stores to validate the elasticity estimate before full rollout. "
                "Reallocate promotional spend from this category to more elastic categories."
            )
        else:
            return (
                f"{category} demand is approximately unit-elastic (ε={elasticity:.2f}): price changes "
                "have a neutral effect on revenue. Focus on volume growth through assortment expansion "
                "and availability improvement rather than price adjustment in this category."
            )
