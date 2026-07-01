"""Customer-level feature engineering for the NovaMart churn prediction model.

Produces one row per customer with RFM metrics, behavioural features,
loyalty programme attributes, and encoded demographics.

Target variable: is_churned (bool) — no purchase in the last 90 days of the data window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class CustomerFeatureEngineer:
    """Build a customer feature matrix for churn prediction modelling.

    Usage::

        from bcgx.features.customer_features import CustomerFeatureEngineer

        engineer = CustomerFeatureEngineer()
        features = engineer.build(transactions, customers)
    """

    CHURN_WINDOW_DAYS: int = 90  # no purchase in this window → churned

    def build(
        self,
        transactions: pd.DataFrame,
        customers: pd.DataFrame,
        reference_date: str | None = None,
    ) -> pd.DataFrame:
        """Construct the customer feature matrix.

        Args:
            transactions: Transaction-level data with customer_id, date, gross_revenue,
                gross_profit, discount_pct, product_id.
            customers: Customer master with customer_id, loyalty_tier, acquisition_channel,
                acquisition_date, age_group, segment.
            reference_date: ISO date string for recency calculation; defaults to max transaction date.

        Returns:
            DataFrame with one row per customer and feature columns.
            Index: customer_id.
        """
        logger.info(
            f"Building customer feature matrix for {customers['customer_id'].nunique():,} customers"
        )

        tx = transactions.copy()
        tx["date"] = pd.to_datetime(tx["date"])

        ref = pd.Timestamp(reference_date) if reference_date is not None else tx["date"].max()

        # --- RFM features ---
        rfm = (
            tx.groupby("customer_id")
            .agg(
                last_purchase_date=("date", "max"),
                total_transactions=("transaction_id", "count"),
                total_spend=("gross_revenue", "sum"),
                total_profit=("gross_profit", "sum"),
                n_distinct_products=("product_id", "nunique"),
            )
            .reset_index()
        )
        rfm["days_since_last_purchase"] = (ref - rfm["last_purchase_date"]).dt.days
        rfm["avg_basket_size"] = rfm["total_spend"] / rfm["total_transactions"].clip(lower=1)

        # --- RFM scores (quintiles 1-5, 5=best) ---
        rfm["recency_score"] = self._quintile_score(rfm["days_since_last_purchase"], reverse=True)
        rfm["frequency_score"] = self._quintile_score(rfm["total_transactions"])
        rfm["monetary_score"] = self._quintile_score(rfm["total_spend"])

        # --- Discount sensitivity ---
        discount_sens = (
            tx.groupby("customer_id")["discount_pct"]
            .mean()
            .reset_index(name="discount_sensitivity")
        )

        # --- Preferred category (most frequent by transaction count) ---
        if "product_id" in tx.columns:
            # Use store_format as category proxy if category not available in transactions
            # (transactions don't include category directly; we use product_id)
            pref_cat = (
                tx.groupby("customer_id")["store_format"]
                .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown")
                .reset_index(name="preferred_store_format")
            )
        else:
            pref_cat = pd.DataFrame({"customer_id": rfm["customer_id"], "preferred_store_format": "unknown"})

        # --- Customer master features ---
        cust = customers.copy()
        cust["acquisition_date"] = pd.to_datetime(cust["acquisition_date"], errors="coerce")
        cust["months_as_customer"] = (
            (ref.year - cust["acquisition_date"].dt.year) * 12
            + (ref.month - cust["acquisition_date"].dt.month)
        ).clip(lower=0)

        # Encode loyalty tier ordinally
        tier_map = {"Bronze": 0, "Silver": 1, "Gold": 2}
        cust["loyalty_tier_code"] = cust["loyalty_tier"].map(tier_map).fillna(0).astype(int)

        # Encode acquisition channel
        acq_channel_dummies = pd.get_dummies(cust["acquisition_channel"], prefix="acq", dtype=int)
        age_group_dummies = pd.get_dummies(cust["age_group"], prefix="age", dtype=int)

        meta_cols = ["customer_id", "loyalty_tier_code", "months_as_customer"]
        cust_meta = pd.concat(
            [
                cust[meta_cols].reset_index(drop=True),
                acq_channel_dummies,
                age_group_dummies,
            ],
            axis=1,
        )

        # --- Merge all feature groups ---
        features = (
            rfm.merge(discount_sens, on="customer_id", how="left")
            .merge(pref_cat, on="customer_id", how="left")
            .merge(cust_meta, on="customer_id", how="left")
        )

        # Drop non-feature columns
        drop_cols = ["last_purchase_date"]
        features = features.drop(columns=[c for c in drop_cols if c in features.columns])

        # --- Target: is_churned ---
        features["is_churned"] = features["days_since_last_purchase"] >= self.CHURN_WINDOW_DAYS

        # --- Fill NAs ---
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0.0)

        object_cols = features.select_dtypes(include=["object", "category"]).columns
        object_cols = [c for c in object_cols if c != "customer_id"]
        features[object_cols] = features[object_cols].fillna("unknown")

        features = features.set_index("customer_id")
        logger.info(
            f"Customer feature matrix built: {features.shape[0]:,} customers × {features.shape[1]} features"
        )
        return features

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _quintile_score(series: pd.Series, reverse: bool = False) -> pd.Series:  # type: ignore[type-arg]
        """Assign quintile score 1-5.  If reverse=True, lower values get score 5 (e.g. recency)."""
        labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
        try:
            return pd.qcut(series, q=5, labels=labels, duplicates="drop").astype(int)
        except ValueError:
            ranked = series.rank(method="first", ascending=not reverse)
            n = len(ranked)
            return np.ceil(ranked / n * 5).clip(1, 5).astype(int)
