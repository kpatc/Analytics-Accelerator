"""Recommendation Prioritiser — computes RICE scores and ranks recommendations.

RICE scoring formula:
    RICE = (Reach * Impact * Confidence) / Effort

Where:
    Reach     = number of customers or stores affected
    Impact    = subjective impact score (1-10)
    Confidence = model confidence (0-1)
    Effort    = implementation effort in weeks
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from bcgx.recommendations.schema import Recommendation


class RecommendationPrioritizer:
    """Scores and ranks Recommendation objects using the RICE framework.

    Usage::

        prioritizer = RecommendationPrioritizer()
        ranked = prioritizer.prioritize(recommendations)
        df = prioritizer.to_dataframe(ranked)
    """

    def compute_rice_score(self, rec: Recommendation) -> float:
        """Compute RICE score for a single recommendation.

        Formula: Reach × Impact × Confidence / Effort

        Args:
            rec: Recommendation object with reach, impact_score, confidence, effort.

        Returns:
            RICE score as a float (higher = more attractive).
        """
        effort = max(rec.implementation_effort_weeks, 1)  # avoid division by zero
        reach_scaled = rec.reach / 1000.0  # scale reach to thousands for readability
        score = (reach_scaled * rec.impact_score * rec.confidence) / effort
        return round(score, 2)

    def prioritize(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        """Compute RICE scores and return recommendations sorted by RICE descending.

        Args:
            recommendations: List of Recommendation objects (rice_score will be updated).

        Returns:
            New list sorted by rice_score descending.
        """
        scored: list[Recommendation] = []
        for rec in recommendations:
            rec.rice_score = self.compute_rice_score(rec)
            scored.append(rec)

        ranked = sorted(scored, key=lambda r: r.rice_score, reverse=True)
        logger.info(
            f"Prioritised {len(ranked)} recommendations by RICE score. "
            f"Top: {ranked[0].title[:60] if ranked else 'N/A'}"
        )
        return ranked

    def to_dataframe(self, recommendations: list[Recommendation]) -> pd.DataFrame:
        """Convert a list of Recommendations to a pandas DataFrame.

        Args:
            recommendations: List of Recommendation objects.

        Returns:
            DataFrame with one row per recommendation, sorted by rice_score descending.
        """
        rows = self.to_dict_list(recommendations)
        df = pd.DataFrame(rows)
        if not df.empty and "rice_score" in df.columns:
            df = df.sort_values("rice_score", ascending=False).reset_index(drop=True)
        return df

    def to_dict_list(self, recommendations: list[Recommendation]) -> list[dict]:
        """Convert a list of Recommendations to a list of plain dicts.

        Args:
            recommendations: List of Recommendation objects.

        Returns:
            List of dicts suitable for JSON serialisation.
        """
        return [
            {
                "id": r.id,
                "title": r.title,
                "category": r.category,
                "description": r.description,
                "evidence": r.evidence,
                "expected_revenue_impact_usd": round(r.expected_revenue_impact_usd, 2),
                "expected_profit_impact_usd": round(r.expected_profit_impact_usd, 2),
                "confidence": r.confidence,
                "difficulty": r.difficulty.value,
                "implementation_effort_weeks": r.implementation_effort_weeks,
                "timeline": r.timeline.value,
                "risk": r.risk,
                "priority": r.priority.value,
                "roi": round(r.roi, 2),
                "reach": r.reach,
                "impact_score": r.impact_score,
                "rice_score": r.rice_score,
                "dependencies": r.dependencies,
                "kpis_to_track": r.kpis_to_track,
            }
            for r in recommendations
        ]
