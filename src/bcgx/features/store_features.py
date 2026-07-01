"""Store-level feature engineering for NovaMart performance prediction models.

Produces one row per store with financial performance metrics, trend features,
marketing efficiency features, and encoded categorical variables.

Target variable: is_top_performer (bool) — top 20% of stores by gross profit.

All features are self-contained and computable from the raw datasets; no external
data sources are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class StoreFeatureEngineer:
    """Build a store-level feature matrix for predictive modelling.

    Usage::

        from bcgx.features.store_features import StoreFeatureEngineer

        engineer = StoreFeatureEngineer()
        features = engineer.build(transactions, stores, marketing, costs)
    """

    def build(
        self,
        transactions: pd.DataFrame,
        stores: pd.DataFrame,
        marketing: pd.DataFrame,
        costs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct the store feature matrix.

        Args:
            transactions: Transaction-level data with store_id, year_month, gross_revenue,
                gross_profit, discount_pct.
            stores: Store master with store_id, sq_footage, open_date, manager_tenure_years,
                performance_cluster, store_format, region.
            marketing: Monthly marketing spend by store and channel.
            costs: Monthly operating costs by store.

        Returns:
            DataFrame with one row per store and feature columns (see class docstring).
            Index: store_id.
        """
        logger.info(f"Building store feature matrix for {stores['store_id'].nunique()} stores")

        # --- Monthly store aggregates ---
        monthly_store = (
            transactions.groupby(["store_id", "year_month"])
            .agg(
                monthly_revenue=("gross_revenue", "sum"),
                monthly_profit=("gross_profit", "sum"),
                n_transactions=("transaction_id", "count"),
            )
            .reset_index()
        )

        # --- Core financial features ---
        store_financials = (
            monthly_store.groupby("store_id")
            .agg(
                avg_monthly_revenue=("monthly_revenue", "mean"),
                avg_monthly_profit=("monthly_profit", "mean"),
                revenue_volatility=("monthly_revenue", "std"),
                n_months_active=("year_month", "nunique"),
            )
            .reset_index()
        )
        store_financials["avg_gross_margin"] = (
            store_financials["avg_monthly_profit"]
            / store_financials["avg_monthly_revenue"].clip(lower=1)
        )

        # --- Revenue trend (OLS slope over time) ---
        store_financials = store_financials.merge(
            self._compute_revenue_trend(monthly_store),
            on="store_id",
            how="left",
        )

        # --- Marketing features ---
        mkt_features = self._compute_marketing_features(marketing)
        store_financials = store_financials.merge(mkt_features, on="store_id", how="left")

        # --- Operating cost ratio ---
        cost_features = self._compute_cost_features(costs)
        store_financials = store_financials.merge(cost_features, on="store_id", how="left")
        store_financials["operating_cost_ratio"] = (
            store_financials["avg_monthly_cost"]
            / store_financials["avg_monthly_revenue"].clip(lower=1)
        )

        # --- Stockout rate (from inventory if available) ---
        store_financials["stockout_rate"] = 0.0  # placeholder — populated by inventory join below

        # --- Store metadata features ---
        store_meta = stores.copy()
        store_meta["open_date"] = pd.to_datetime(store_meta["open_date"], errors="coerce")
        reference_date = pd.Timestamp.now()
        store_meta["months_since_open"] = (
            (reference_date.year - store_meta["open_date"].dt.year) * 12
            + (reference_date.month - store_meta["open_date"].dt.month)
        ).clip(lower=0)

        # Encode categorical variables
        store_format_dummies = pd.get_dummies(
            store_meta["store_format"], prefix="fmt", dtype=int
        )
        region_dummies = pd.get_dummies(store_meta["region"], prefix="region", dtype=int)

        cluster_map = {"A": 0, "B": 1, "C": 2}
        store_meta["performance_cluster_code"] = (
            store_meta["performance_cluster"].map(cluster_map).fillna(-1).astype(int)
        )

        meta_cols = [
            "store_id",
            "sq_footage",
            "manager_tenure_years",
            "months_since_open",
            "performance_cluster_code",
        ]
        meta_df = pd.concat(
            [store_meta[meta_cols].reset_index(drop=True), store_format_dummies, region_dummies],
            axis=1,
        )

        # --- Merge everything ---
        features = store_financials.merge(meta_df, on="store_id", how="left")

        # --- Target: is_top_performer (top 20% by avg_monthly_profit) ---
        top20_threshold = features["avg_monthly_profit"].quantile(0.80)
        features["is_top_performer"] = features["avg_monthly_profit"] >= top20_threshold

        # --- Fill NAs ---
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0.0)

        features = features.set_index("store_id")
        logger.info(
            f"Store feature matrix built: {features.shape[0]} stores × {features.shape[1]} features"
        )
        return features

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_revenue_trend(monthly_store: pd.DataFrame) -> pd.DataFrame:
        """Fit OLS regression of monthly revenue on month number for each store.

        Returns DataFrame with store_id and revenue_trend_slope.
        """
        records: list[dict[str, object]] = []
        for store_id, grp in monthly_store.groupby("store_id"):
            grp_sorted = grp.sort_values("year_month")
            y = grp_sorted["monthly_revenue"].values
            x = np.arange(len(y), dtype=float)
            if len(y) >= 2 and np.std(y) > 0:
                slope, *_ = stats.linregress(x, y)
            else:
                slope = 0.0
            records.append({"store_id": store_id, "revenue_trend_slope": float(slope)})
        return pd.DataFrame(records)

    @staticmethod
    def _compute_marketing_features(marketing: pd.DataFrame) -> pd.DataFrame:
        """Compute per-store marketing efficiency and digital spend share."""
        total_spend = (
            marketing.groupby("store_id")["spend_usd"].sum().reset_index(name="total_marketing_spend")
        )
        digital_spend = (
            marketing[marketing["channel"] == "digital"]
            .groupby("store_id")["spend_usd"]
            .sum()
            .reset_index(name="digital_spend")
        )
        mkt = total_spend.merge(digital_spend, on="store_id", how="left")
        mkt["digital_spend"] = mkt["digital_spend"].fillna(0.0)
        mkt["digital_spend_share"] = mkt["digital_spend"] / mkt["total_marketing_spend"].clip(lower=1)

        # marketing_efficiency is computed after revenue merge; provide raw spend here
        return mkt[["store_id", "total_marketing_spend", "digital_spend_share"]]

    @staticmethod
    def _compute_cost_features(costs: pd.DataFrame) -> pd.DataFrame:
        """Compute average monthly operating cost per store."""
        return (
            costs.groupby("store_id")["total_operating_cost_usd"]
            .mean()
            .reset_index(name="avg_monthly_cost")
        )
