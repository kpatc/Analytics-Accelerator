"""Price elasticity ML pipeline.

ML-enhanced price elasticity estimation using log-log regression per category
with brand_type interaction terms.

Economic interpretation:
    ln(quantity) = β₀ + β₁ × ln(price) + β₂ × brand_type + β₃ × ln(price) × brand_type + ε
    β₁ = price elasticity (typically < 0 for normal goods)

Answers:
- What is the price elasticity for each product/category?
- What is the optimal price given a target margin?
- How should prices be set to maximise revenue or profit?
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass
class ElasticityResult:
    """Per-product elasticity estimate."""

    product_id: str
    category: str
    elasticity: float  # d(ln Q)/d(ln P) — typically negative
    lower_ci: float  # 95% confidence interval lower bound
    upper_ci: float  # 95% confidence interval upper bound
    r_squared: float
    n_observations: int
    brand_type: str
    avg_selling_price: float
    avg_demand: float


# ---------------------------------------------------------------------------
# Elasticity Model Pipeline
# ---------------------------------------------------------------------------


class ElasticityModelPipeline:
    """ML-enhanced price elasticity model (log-log regression per category).

    Fits one log-log OLS model per product category with brand_type interaction.
    Extracts per-product elasticity estimates from the fitted models.

    Usage::

        pipeline = ElasticityModelPipeline()
        pipeline.fit(transactions, products)
        elasticity_df = pipeline.get_all_elasticities()
        optimal_price = pipeline.get_optimal_price("PROD001", target_margin=0.30)
    """

    def __init__(self, alpha: float = 0.01, min_obs_per_product: int = 5) -> None:
        """
        Args:
            alpha: Ridge regularisation for per-category models.
            min_obs_per_product: Minimum number of observations to estimate elasticity.
        """
        self._alpha = alpha
        self._min_obs = min_obs_per_product
        self._category_models: dict[str, Ridge] = {}
        self._product_elasticities: dict[str, ElasticityResult] = {}
        self._product_meta: dict[str, dict[str, object]] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        transactions: pd.DataFrame,
        products: pd.DataFrame,
    ) -> None:
        """Fit log-log price elasticity models per category.

        Args:
            transactions: Transaction data with product_id, unit_price, quantity,
                          gross_revenue, gross_profit, discount_pct.
            products: Product catalogue with product_id, category, brand_type,
                      unit_cost, list_price.
        """
        logger.info(
            f"Fitting elasticity models on {len(transactions):,} transactions"
            f" × {products['product_id'].nunique():,} products"
        )

        # Build product-level time series of (price, quantity) pairs
        tx = transactions.copy()
        tx["log_price"] = np.log(tx["unit_price"].clip(lower=0.01))
        tx["log_quantity"] = np.log(tx["quantity"].clip(lower=0.001))

        # Merge with product metadata
        prod_meta = products[["product_id", "category", "brand_type", "unit_cost", "list_price"]].copy()
        tx = tx.merge(prod_meta, on="product_id", how="left")
        tx["brand_type"] = tx["brand_type"].fillna("national")

        # Brand type indicator (national=0, private_label=1, premium=2)
        brand_map = {"national": 0, "private_label": 1, "premium": 2}
        tx["brand_code"] = tx["brand_type"].map(brand_map).fillna(0).astype(int)

        # Per-product summary stats for prediction
        prod_agg = (
            tx.groupby("product_id").agg(
                avg_price=("unit_price", "mean"),
                avg_quantity=("quantity", "mean"),
                n_obs=("quantity", "count"),
            )
        )

        # Fit per-category models
        self._category_models = {}
        self._product_elasticities = {}

        categories = tx["category"].dropna().unique()
        for cat in categories:
            try:
                cat_df = tx[tx["category"] == cat].dropna(
                    subset=["log_price", "log_quantity"]
                )
                if len(cat_df) < 10:
                    continue
                self._fit_category_model(cat, cat_df, prod_agg, prod_meta)
            except Exception as exc:
                logger.debug(f"Category '{cat}' model failed: {exc}")

        # Store product metadata for prediction
        for _, row in prod_meta.iterrows():
            pid = str(row["product_id"])
            self._product_meta[pid] = {
                "category": str(row.get("category", "unknown")),
                "brand_type": str(row.get("brand_type", "national")),
                "unit_cost": float(row.get("unit_cost", 0.0)),
                "list_price": float(row.get("list_price", 0.0)),
            }

        self._is_fitted = True
        logger.info(
            f"Elasticity models fitted: {len(self._category_models)} categories"
            f" | {len(self._product_elasticities)} products with estimates"
        )

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #

    def predict_demand(self, product_id: str, price: float) -> float:
        """Predict demand (quantity) at a given price.

        Uses the product's category elasticity model with log-log transformation.

        Args:
            product_id: Product identifier.
            price: Unit price to evaluate.

        Returns:
            Predicted demand (quantity units).
        """
        self._check_fitted()
        if product_id not in self._product_elasticities:
            raise ValueError(f"No elasticity estimate for product '{product_id}'")

        result = self._product_elasticities[product_id]
        # ln(Q) = ln(Q_avg) + elasticity × ln(P / P_avg)
        if result.avg_selling_price <= 0 or result.avg_demand <= 0:
            return result.avg_demand
        ln_quantity = (
            np.log(result.avg_demand)
            + result.elasticity * np.log(max(price, 0.01) / result.avg_selling_price)
        )
        return float(np.exp(ln_quantity))

    def get_optimal_price(
        self,
        product_id: str,
        target_margin: float = 0.30,
    ) -> float:
        """Find the revenue-maximising price that meets a target gross margin.

        The optimal revenue price is P* = unit_cost / (1 - target_margin) for
        margin constraint. We also compute the unconstrained revenue-maximising
        price and return whichever is higher (the binding constraint).

        Args:
            product_id: Product identifier.
            target_margin: Required gross margin fraction (0-1).

        Returns:
            Optimal price in USD.
        """
        self._check_fitted()
        if product_id not in self._product_meta:
            raise ValueError(f"No metadata for product '{product_id}'")

        meta = self._product_meta[product_id]
        unit_cost = float(meta.get("unit_cost", 0.0))

        # Minimum price to achieve target margin
        if unit_cost > 0:
            min_price = unit_cost / max(1.0 - target_margin, 0.01)
        else:
            min_price = 0.01

        if product_id not in self._product_elasticities:
            return min_price

        result = self._product_elasticities[product_id]
        elasticity = result.elasticity

        # Revenue-maximising price: P* = P_avg × (ε / (ε + 1))  [only if ε < -1]
        if elasticity < -1:
            rev_max_price = result.avg_selling_price * (elasticity / (elasticity + 1))
            optimal = max(min_price, rev_max_price)
        else:
            # Inelastic — margin constraint is binding
            optimal = min_price

        return float(max(optimal, min_price))

    def get_all_elasticities(self) -> pd.DataFrame:
        """Return a DataFrame of per-product elasticity estimates.

        Returns:
            DataFrame with columns:
            product_id, category, elasticity, lower_ci, upper_ci, r_squared,
            n_observations, brand_type, avg_selling_price.
        """
        self._check_fitted()
        records = []
        for pid, result in self._product_elasticities.items():
            records.append(
                {
                    "product_id": pid,
                    "category": result.category,
                    "elasticity": result.elasticity,
                    "lower_ci": result.lower_ci,
                    "upper_ci": result.upper_ci,
                    "r_squared": result.r_squared,
                    "n_observations": result.n_observations,
                    "brand_type": result.brand_type,
                    "avg_selling_price": result.avg_selling_price,
                }
            )
        if not records:
            return pd.DataFrame(
                columns=[
                    "product_id",
                    "category",
                    "elasticity",
                    "lower_ci",
                    "upper_ci",
                    "r_squared",
                    "n_observations",
                    "brand_type",
                    "avg_selling_price",
                ]
            )
        df = pd.DataFrame(records)
        df = df.sort_values("elasticity")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _fit_category_model(
        self,
        category: str,
        cat_df: pd.DataFrame,
        prod_agg: pd.DataFrame,
        prod_meta: pd.DataFrame,
    ) -> None:
        """Fit a log-log Ridge model for one category and extract elasticities."""
        # Feature matrix: log_price, brand_code, interaction (log_price × brand_code)
        X = np.column_stack(
            [
                cat_df["log_price"].values,
                cat_df["brand_code"].values,
                cat_df["log_price"].values * cat_df["brand_code"].values,
            ]
        )
        y = cat_df["log_quantity"].values

        model = Ridge(alpha=self._alpha, fit_intercept=True)
        model.fit(X, y)
        self._category_models[category] = model

        # Extract per-product elasticities
        cat_products = cat_df["product_id"].unique()
        for pid in cat_products:
            pid_str = str(pid)
            prod_df = cat_df[cat_df["product_id"] == pid]
            n_obs = len(prod_df)

            if n_obs < self._min_obs:
                continue

            if pid_str not in prod_agg.index.astype(str).tolist():
                avg_price = float(prod_df["unit_price"].mean())
                avg_qty = float(prod_df["quantity"].mean())
            else:
                try:
                    agg_row = prod_agg.loc[pid]
                    avg_price = float(agg_row["avg_price"])
                    avg_qty = float(agg_row["avg_quantity"])
                except Exception:
                    avg_price = float(prod_df["unit_price"].mean())
                    avg_qty = float(prod_df["quantity"].mean())

            # Get brand code for this product
            brand_code_val = int(prod_df["brand_code"].mode().iloc[0])
            brand_type_val = str(prod_df["brand_type"].mode().iloc[0])

            # Elasticity at this product: β_price + β_interaction × brand_code
            coef = model.coef_
            elasticity = float(coef[0] + coef[2] * brand_code_val)

            # R² for this product's own data
            X_prod = np.column_stack(
                [
                    prod_df["log_price"].values,
                    prod_df["brand_code"].values,
                    prod_df["log_price"].values * prod_df["brand_code"].values,
                ]
            )
            y_prod = prod_df["log_quantity"].values
            y_pred = model.predict(X_prod)
            ss_res = np.sum((y_prod - y_pred) ** 2)
            ss_tot = np.sum((y_prod - y_prod.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Bootstrap CI (simplified: ±1 SE of elasticity across observations)
            if n_obs >= 10:
                price_std = float(prod_df["log_price"].std())
                qty_std = float(prod_df["log_quantity"].std())
                se = qty_std / max(price_std * np.sqrt(n_obs), 1e-8)
                lower_ci = elasticity - 1.96 * se
                upper_ci = elasticity + 1.96 * se
            else:
                lower_ci = elasticity - 0.5
                upper_ci = elasticity + 0.5

            self._product_elasticities[pid_str] = ElasticityResult(
                product_id=pid_str,
                category=category,
                elasticity=elasticity,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                r_squared=float(r2),
                n_observations=n_obs,
                brand_type=brand_type_val,
                avg_selling_price=avg_price,
                avg_demand=avg_qty,
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ElasticityModelPipeline must be fitted before calling this method."
            )
