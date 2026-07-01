"""Product-level feature engineering for NovaMart price elasticity and product performance models.

Produces one row per product with pricing, volume, and category-relative performance metrics.

Key features:
- Pricing metrics: avg_selling_price, avg_discount_pct, price_variability
- Volume and revenue: total_units_sold, total_revenue, total_profit
- Margin: gross_margin_pct
- Category context: revenue_share_in_category
- Brand type encoding: is_private_label, brand_type_code
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class ProductFeatureEngineer:
    """Build a product feature matrix for pricing and elasticity models.

    Usage::

        from bcgx.features.product_features import ProductFeatureEngineer

        engineer = ProductFeatureEngineer()
        features = engineer.build(transactions, products)
    """

    def build(
        self,
        transactions: pd.DataFrame,
        products: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct the product feature matrix.

        Args:
            transactions: Transaction-level data with product_id, quantity,
                unit_price, discount_pct, gross_revenue, gross_profit.
            products: Product catalogue with product_id, category, subcategory,
                brand_type, unit_cost, list_price, gross_margin_pct.

        Returns:
            DataFrame with one row per product and feature columns.
            Index: product_id.
        """
        logger.info(f"Building product feature matrix for {products['product_id'].nunique():,} products")

        tx = transactions.copy()

        # --- Core transaction aggregates per product ---
        prod_agg = (
            tx.groupby("product_id")
            .agg(
                total_units_sold=("quantity", "sum"),
                total_revenue=("gross_revenue", "sum"),
                total_profit=("gross_profit", "sum"),
                n_transactions=("transaction_id", "count"),
                avg_selling_price=("unit_price", "mean"),
                avg_discount_pct=("discount_pct", "mean"),
                price_variability=("unit_price", "std"),
            )
            .reset_index()
        )

        prod_agg["gross_margin_pct"] = (
            prod_agg["total_profit"] / prod_agg["total_revenue"].clip(lower=1) * 100
        )

        # --- Merge with product catalogue ---
        prod_meta = products[
            ["product_id", "category", "subcategory", "brand_type", "unit_cost", "list_price"]
        ].copy()

        features = prod_agg.merge(prod_meta, on="product_id", how="left")

        # --- Category-relative metrics ---
        cat_revenue = (
            features.groupby("category")["total_revenue"]
            .transform("sum")
            .clip(lower=1)
        )
        features["revenue_share_in_category"] = features["total_revenue"] / cat_revenue * 100

        # --- Brand type encoding ---
        features["is_private_label"] = (features["brand_type"] == "private_label").astype(int)

        brand_type_map: dict[str, int] = {}
        for i, bt in enumerate(sorted(features["brand_type"].dropna().unique())):
            brand_type_map[bt] = i
        features["brand_type_code"] = features["brand_type"].map(brand_type_map).fillna(-1).astype(int)

        # --- Category encoding ---
        category_dummies = pd.get_dummies(features["category"], prefix="cat", dtype=int)
        features = pd.concat([features, category_dummies], axis=1)

        # --- Fill NAs ---
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0.0)

        features = features.set_index("product_id")
        logger.info(
            f"Product feature matrix built: {features.shape[0]:,} products × {features.shape[1]} features"
        )
        return features
