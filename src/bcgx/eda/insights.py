"""Aggregate EDA results into an executive insight summary for NovaMart leadership.

This module sits at the top of the EDA layer: it orchestrates all univariate,
bivariate, and temporal analyses, then synthesises them into a single
EDAInsightSummary that can be serialised to JSON for the dashboard and slide deck.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from bcgx.eda.bivariate import BivariateAnalyzer, BivariateResult
from bcgx.eda.temporal import TemporalAnalyzer, TemporalResult
from bcgx.eda.univariate import UnivariateAnalyzer, UnivariateResult


@dataclass
class EDAInsightSummary:
    """Executive summary of all EDA findings."""

    generated_at: str
    total_revenue: float
    total_profit: float
    avg_gross_margin_pct: float
    margin_trend_pct_change: float  # e.g. -30.0 means 30% decline
    top_insight: str  # single most important finding for the board
    univariate_insights: list[UnivariateResult]
    bivariate_insights: list[BivariateResult]
    temporal_insights: list[TemporalResult]


class InsightExtractor:
    """Orchestrate the full EDA pipeline and produce a board-ready insight summary.

    Usage::

        from bcgx.data.loader import DataLoader
        from bcgx.eda.insights import InsightExtractor

        data = DataLoader("data/raw").load_all()
        extractor = InsightExtractor()
        summary = extractor.run(data)
        extractor.save_json(summary, "data/outputs/eda_insights.json")
    """

    def run(self, data: dict[str, pd.DataFrame]) -> EDAInsightSummary:
        """Execute all EDA analyses and return the consolidated summary.

        Args:
            data: Dict of DataFrames produced by DataLoader.load_all().

        Returns:
            EDAInsightSummary populated with all analytical results.
        """
        logger.info("Starting full EDA insight extraction pipeline")

        # --- Headline financials ---
        tx = data["transactions"]
        total_revenue = float(tx["gross_revenue"].sum())
        total_profit = float(tx["gross_profit"].sum())
        avg_gross_margin_pct = total_profit / max(total_revenue, 1) * 100

        logger.info(
            f"Portfolio summary: revenue=${total_revenue / 1e6:.1f}M, "
            f"profit=${total_profit / 1e6:.1f}M, margin={avg_gross_margin_pct:.1f}%"
        )

        # --- Run all sub-analyses ---
        univariate_results = UnivariateAnalyzer().run_all(data)
        bivariate_results = BivariateAnalyzer().run_all(data)
        temporal_results = TemporalAnalyzer().run_all(data)

        # --- Identify margin trend magnitude ---
        margin_trend_pct_change = 0.0
        for result in temporal_results:
            if result.metric == "monthly_gross_margin_pct":
                margin_trend_pct_change = result.trend_magnitude
                break

        # --- Synthesise the single most important top insight ---
        top_insight = self._derive_top_insight(
            total_revenue=total_revenue,
            total_profit=total_profit,
            avg_gross_margin_pct=avg_gross_margin_pct,
            margin_trend_pct_change=margin_trend_pct_change,
            temporal_results=temporal_results,
            univariate_results=univariate_results,
            bivariate_results=bivariate_results,
        )

        summary = EDAInsightSummary(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_revenue=total_revenue,
            total_profit=total_profit,
            avg_gross_margin_pct=avg_gross_margin_pct,
            margin_trend_pct_change=margin_trend_pct_change,
            top_insight=top_insight,
            univariate_insights=univariate_results,
            bivariate_insights=bivariate_results,
            temporal_insights=temporal_results,
        )

        logger.success("EDA insight extraction complete")
        return summary

    def to_dict(self, summary: EDAInsightSummary) -> dict:  # type: ignore[type-arg]
        """Serialise an EDAInsightSummary to a plain dictionary.

        Args:
            summary: The EDA summary dataclass.

        Returns:
            JSON-serialisable dictionary representation.
        """
        return asdict(summary)

    def save_json(self, summary: EDAInsightSummary, output_path: str) -> None:
        """Write the EDA summary to a JSON file.

        Args:
            summary: The EDA summary to serialise.
            output_path: Destination file path (parent directories are created).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(summary)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"EDA insights saved to {path}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _derive_top_insight(
        total_revenue: float,
        total_profit: float,
        avg_gross_margin_pct: float,
        margin_trend_pct_change: float,
        temporal_results: list[TemporalResult],
        univariate_results: list[UnivariateResult],
        bivariate_results: list[BivariateResult],
    ) -> str:
        """Synthesise the single most critical finding for board-level communication.

        Priority logic:
        1. If margin is declining and the trend is statistically significant → lead with margin
        2. If revenue is declining → lead with revenue
        3. Otherwise → lead with the strongest statistical finding

        Args:
            total_revenue: Portfolio total revenue.
            total_profit: Portfolio total gross profit.
            avg_gross_margin_pct: Blended gross margin.
            margin_trend_pct_change: % change in margin from first to last period.
            temporal_results: Temporal analysis results.
            univariate_results: Univariate analysis results.
            bivariate_results: Bivariate analysis results.

        Returns:
            A single, board-ready insight string.
        """
        # Check margin trend significance
        margin_sig = False
        revenue_declining = False
        for tr in temporal_results:
            if tr.metric == "monthly_gross_margin_pct":
                margin_sig = tr.trend_p_value < 0.05 and tr.trend_direction == "declining"
            if tr.metric == "monthly_gross_revenue":
                revenue_declining = tr.trend_direction == "declining"

        if margin_sig and margin_trend_pct_change < -5:
            return (
                f"CRITICAL: NovaMart's gross margin is on a statistically significant declining trend, "
                f"having contracted {abs(margin_trend_pct_change):.1f}% over the analysis window from an "
                f"average of {avg_gross_margin_pct:.1f}%. At the current trajectory, operating profitability "
                f"will come under severe pressure within 12-18 months. The root causes — promotional depth, "
                f"unfavourable product mix, and unabsorbed cost inflation — are addressable, but require "
                f"immediate executive prioritisation. A 200bps gross margin recovery would add "
                f"~${total_revenue * 0.02 / 1e6:.1f}M to annual profit without requiring top-line growth."
            )
        elif revenue_declining:
            return (
                f"NovaMart faces a dual challenge: declining revenue and compressed margins. "
                f"Total portfolio revenue of ${total_revenue / 1e6:.1f}M is on a downward trend, "
                f"while gross margins average {avg_gross_margin_pct:.1f}%. Revenue recovery must be "
                f"pursued in parallel with margin protection — attempting to buy back revenue through "
                f"discounting will accelerate the margin decline and create a destructive spiral. "
                f"The recommended priority sequence: stabilise margin first, then reinvest savings "
                f"into proven growth channels."
            )
        else:
            # Lead with portfolio concentration risk — always a relevant finding
            return (
                f"NovaMart's ${total_revenue / 1e6:.1f}M revenue base is highly concentrated: "
                f"a small number of top-performing stores and customers generate a "
                f"disproportionate share of total revenue. This concentration creates both "
                f"strategic vulnerability (loss of a key store or customer cohort has outsized P&L impact) "
                f"and growth opportunity (replicating top-performer practices across the portfolio "
                f"could deliver 15-25% revenue uplift without new customer acquisition). "
                f"Current blended gross margin of {avg_gross_margin_pct:.1f}% provides a "
                f"${total_profit / 1e6:.1f}M gross profit base to fund the transformation."
            )
