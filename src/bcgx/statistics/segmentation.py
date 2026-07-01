"""Customer and store segmentation statistics for NovaMart.

Provides:
- Store cluster profiling (revenue, profit, margin by cluster)
- Customer segment profiling (by loyalty tier / segment)
- RFM scoring with quintile-based scores and standard segment labels

No plotting — all outputs are structured data objects for the dashboard and
reporting layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SegmentProfile:
    """Business profile for a single store or customer segment."""

    segment_name: str
    segment_type: str  # "customer" | "store"
    n_members: int
    pct_of_total: float
    avg_revenue: float
    avg_profit: float
    avg_margin_pct: float
    revenue_share_pct: float
    profit_share_pct: float
    key_characteristic: str  # business description for slide decks


@dataclass
class RFMSegment:
    """RFM score and segment label for a single customer."""

    customer_id: str
    recency_score: int  # 1-5 (5 = most recent)
    frequency_score: int  # 1-5 (5 = highest frequency)
    monetary_score: int  # 1-5 (5 = highest spend)
    rfm_score: str  # e.g. "555"
    segment_label: str  # "Champions" | "Loyal Customers" | "At Risk" | "Lost" | etc.


class SegmentationAnalyzer:
    """Compute store and customer segmentation profiles and RFM scores.

    All methods return pure data structures (dataclasses or DataFrames);
    no plotting is performed.
    """

    # ------------------------------------------------------------------ #
    # Store cluster profiling                                              #
    # ------------------------------------------------------------------ #

    def profile_store_clusters(
        self, transactions: pd.DataFrame, stores: pd.DataFrame
    ) -> list[SegmentProfile]:
        """Profile each store performance cluster (A, B, C) by revenue and profit.

        Args:
            transactions: Transaction data with store_id, gross_revenue, gross_profit.
            stores: Store master with performance_cluster.

        Returns:
            List of SegmentProfile, one per cluster.
        """
        logger.info("Profiling store clusters")

        store_perf = (
            transactions.groupby("store_id")
            .agg(
                total_revenue=("gross_revenue", "sum"),
                total_profit=("gross_profit", "sum"),
            )
            .reset_index()
        )
        store_perf["margin_pct"] = (
            store_perf["total_profit"] / store_perf["total_revenue"].clip(lower=1) * 100
        )
        store_perf = store_perf.merge(
            stores[["store_id", "performance_cluster"]], on="store_id", how="left"
        )

        total_stores = len(store_perf)
        total_revenue = float(store_perf["total_revenue"].sum())
        total_profit = float(store_perf["total_profit"].sum())

        profiles: list[SegmentProfile] = []
        cluster_descriptions = {
            "A": (
                "Top-performing stores: above-median revenue, above-median margins, and "
                "disproportionate contribution to portfolio profit. These are NovaMart's 'crown jewel' "
                "locations that should receive protected investment and dedicated operational support."
            ),
            "B": (
                "Mid-tier stores with solid revenue but margin compression relative to Cluster A. "
                "These stores represent the highest-ROI improvement opportunity — targeted "
                "interventions in commercial execution and cost management can close the gap to Cluster A."
            ),
            "C": (
                "Underperforming stores with below-average revenue and margins. Some may face "
                "structural disadvantages (location, format, demographics) requiring strategic "
                "decisions on investment, format change, or closure. Prioritise the top third of "
                "Cluster C for a focused turnaround programme."
            ),
        }

        for cluster in sorted(store_perf["performance_cluster"].dropna().unique()):
            grp = store_perf[store_perf["performance_cluster"] == cluster]
            n = len(grp)
            avg_rev = float(grp["total_revenue"].mean())
            avg_profit = float(grp["total_profit"].mean())
            avg_margin = float(grp["margin_pct"].mean())
            rev_share = float(grp["total_revenue"].sum() / max(total_revenue, 1) * 100)
            profit_share = float(grp["total_profit"].sum() / max(total_profit, 1) * 100)
            pct_of_total = float(n / max(total_stores, 1) * 100)

            profiles.append(
                SegmentProfile(
                    segment_name=f"Cluster {cluster}",
                    segment_type="store",
                    n_members=n,
                    pct_of_total=round(pct_of_total, 2),
                    avg_revenue=round(avg_rev, 2),
                    avg_profit=round(avg_profit, 2),
                    avg_margin_pct=round(avg_margin, 2),
                    revenue_share_pct=round(rev_share, 2),
                    profit_share_pct=round(profit_share, 2),
                    key_characteristic=cluster_descriptions.get(str(cluster), ""),
                )
            )

        logger.info(f"Store cluster profiling complete: {len(profiles)} clusters")
        return profiles

    # ------------------------------------------------------------------ #
    # Customer segment profiling                                           #
    # ------------------------------------------------------------------ #

    def profile_customer_segments(
        self, transactions: pd.DataFrame, customers: pd.DataFrame
    ) -> list[SegmentProfile]:
        """Profile each customer loyalty tier by spend and profitability.

        Args:
            transactions: Transaction data with customer_id, gross_revenue, gross_profit.
            customers: Customer master with loyalty_tier.

        Returns:
            List of SegmentProfile, one per loyalty tier.
        """
        logger.info("Profiling customer segments by loyalty tier")

        cust_perf = (
            transactions.groupby("customer_id")
            .agg(
                total_revenue=("gross_revenue", "sum"),
                total_profit=("gross_profit", "sum"),
                n_transactions=("transaction_id", "count"),
            )
            .reset_index()
        )
        cust_perf["margin_pct"] = (
            cust_perf["total_profit"] / cust_perf["total_revenue"].clip(lower=1) * 100
        )
        cust_perf = cust_perf.merge(
            customers[["customer_id", "loyalty_tier"]], on="customer_id", how="left"
        )

        total_customers = len(cust_perf)
        total_revenue = float(cust_perf["total_revenue"].sum())
        total_profit = float(cust_perf["total_profit"].sum())

        tier_descriptions = {
            "Gold": (
                "Highest-value customers: frequent purchasers with the largest basket sizes and "
                "lowest churn risk. Gold-tier customers are NovaMart's most strategically important "
                "cohort — their loyalty must be protected through exclusive benefits and proactive "
                "relationship management."
            ),
            "Silver": (
                "Mid-value customers with strong purchase frequency and moderate basket size. "
                "Silver is the largest loyalty tier by count and represents the primary growth "
                "opportunity — converting Silver customers to Gold behaviour would significantly "
                "increase lifetime value. This tier was most impacted by the recent loyalty fee change."
            ),
            "Bronze": (
                "Entry-tier customers with lower purchase frequency and basket size. Many Bronze "
                "customers are early in their relationship with NovaMart and represent a pipeline "
                "for tier progression. The primary objective is driving repeat purchase occasions "
                "to build habitual engagement."
            ),
        }

        profiles: list[SegmentProfile] = []
        for tier in ["Gold", "Silver", "Bronze"]:
            grp = cust_perf[cust_perf["loyalty_tier"] == tier]
            if grp.empty:
                continue
            n = len(grp)
            avg_rev = float(grp["total_revenue"].mean())
            avg_profit = float(grp["total_profit"].mean())
            avg_margin = float(grp["margin_pct"].mean())
            rev_share = float(grp["total_revenue"].sum() / max(total_revenue, 1) * 100)
            profit_share = float(grp["total_profit"].sum() / max(total_profit, 1) * 100)
            pct_of_total = float(n / max(total_customers, 1) * 100)

            profiles.append(
                SegmentProfile(
                    segment_name=tier,
                    segment_type="customer",
                    n_members=n,
                    pct_of_total=round(pct_of_total, 2),
                    avg_revenue=round(avg_rev, 2),
                    avg_profit=round(avg_profit, 2),
                    avg_margin_pct=round(avg_margin, 2),
                    revenue_share_pct=round(rev_share, 2),
                    profit_share_pct=round(profit_share, 2),
                    key_characteristic=tier_descriptions.get(tier, ""),
                )
            )

        logger.info(f"Customer segment profiling complete: {len(profiles)} segments")
        return profiles

    # ------------------------------------------------------------------ #
    # RFM scoring                                                          #
    # ------------------------------------------------------------------ #

    def compute_rfm(
        self, transactions: pd.DataFrame, reference_date: str | None = None
    ) -> pd.DataFrame:
        """Compute RFM scores for all customers using quintile-based scoring.

        Scoring convention (1=worst, 5=best):
        - Recency: lower days_since_purchase = better (5 = most recent)
        - Frequency: higher transaction count = better (5 = most frequent)
        - Monetary: higher total spend = better (5 = highest spend)

        Standard segment labels:
        - 555: Champions
        - High R + High FM (4-5 across all): Loyal Customers
        - High R, Low FM: Recent Customers
        - Low R, High FM: At Risk
        - All-low: Lost
        - Mid-range: Potential Loyalists / Needs Attention

        Args:
            transactions: Transaction data with customer_id, date, gross_revenue.
            reference_date: ISO date string for recency calculation; defaults to max date.

        Returns:
            DataFrame with one row per customer and columns:
            customer_id, recency_days, frequency, monetary, recency_score,
            frequency_score, monetary_score, rfm_score, segment_label.
        """
        logger.info("Computing RFM scores")

        tx = transactions.copy()
        tx["date"] = pd.to_datetime(tx["date"])

        if reference_date is not None:
            ref = pd.Timestamp(reference_date)
        else:
            ref = tx["date"].max()

        rfm = (
            tx.groupby("customer_id")
            .agg(
                recency_days=("date", lambda x: (ref - x.max()).days),
                frequency=("transaction_id", "count"),
                monetary=("gross_revenue", "sum"),
            )
            .reset_index()
        )

        # Quintile scoring — recency: lower = better so labels are reversed
        def _quintile_score(series: pd.Series, reverse: bool = False) -> pd.Series:  # type: ignore[type-arg]
            """Assign quintile score 1-5.  If reverse=True, lower values get score 5."""
            labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
            try:
                return pd.qcut(series, q=5, labels=labels, duplicates="drop").astype(int)
            except ValueError:
                # Fallback: simple rank-based scoring when there are too few unique values
                ranked = series.rank(method="first", ascending=not reverse)
                n = len(ranked)
                return np.ceil(ranked / n * 5).clip(1, 5).astype(int)

        rfm["recency_score"] = _quintile_score(rfm["recency_days"], reverse=True)
        rfm["frequency_score"] = _quintile_score(rfm["frequency"])
        rfm["monetary_score"] = _quintile_score(rfm["monetary"])

        rfm["rfm_score"] = (
            rfm["recency_score"].astype(str)
            + rfm["frequency_score"].astype(str)
            + rfm["monetary_score"].astype(str)
        )

        rfm["segment_label"] = rfm.apply(self._assign_segment_label, axis=1)

        logger.info(f"RFM scoring complete: {len(rfm):,} customers scored")
        return rfm

    def run_all(self, data: dict[str, pd.DataFrame]) -> dict[str, list]:  # type: ignore[type-arg]
        """Run all segmentation analyses.

        Args:
            data: Dict produced by DataLoader.load_all().

        Returns:
            Dict with keys "store_profiles", "customer_profiles", "rfm_summary".
        """
        logger.info("Running all segmentation analyses")

        store_profiles = self.profile_store_clusters(data["transactions"], data["stores"])
        customer_profiles = self.profile_customer_segments(data["transactions"], data["customers"])
        rfm_df = self.compute_rfm(data["transactions"])

        rfm_summary = (
            rfm_df.groupby("segment_label")
            .agg(
                n_customers=("customer_id", "count"),
                avg_monetary=("monetary", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_recency_days=("recency_days", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        )

        logger.success("Segmentation analysis complete")
        return {
            "store_profiles": store_profiles,
            "customer_profiles": customer_profiles,
            "rfm_summary": rfm_summary,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assign_segment_label(row: pd.Series) -> str:  # type: ignore[type-arg]
        """Map RFM scores to business segment labels.

        Segment rules (heuristic BCG/industry standard):
        - Champions: R=5, F=5, M=5
        - Loyal Customers: R≥4, F≥4, M≥4
        - Potential Loyalists: R≥4, F≥2, M≥2 (high recency, developing frequency)
        - Recent Customers: R≥4, F≤2, M≤2
        - At Risk: R≤2, F≥4, M≥3 (previously valuable, now lapsing)
        - Lost: R=1, F≤2 (no longer engaging)
        - Needs Attention: everything else mid-range
        """
        r = int(row["recency_score"])
        f = int(row["frequency_score"])
        m = int(row["monetary_score"])

        if r == 5 and f == 5 and m == 5:
            return "Champions"
        elif r >= 4 and f >= 4 and m >= 4:
            return "Loyal Customers"
        elif r >= 4 and f >= 2 and m >= 2:
            return "Potential Loyalists"
        elif r >= 4 and f <= 2 and m <= 2:
            return "Recent Customers"
        elif r <= 2 and f >= 4 and m >= 3:
            return "At Risk"
        elif r == 1 and f <= 2:
            return "Lost"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Promising"
        else:
            return "Needs Attention"
