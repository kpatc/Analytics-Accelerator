"""Business-aware churn model evaluator.

Translates raw model predictions into consulting-ready insights:
- Revenue at risk
- High-confidence churner segments
- Churn rate by loyalty tier
- Recommended customer retention interventions
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class ChurnEvaluationReport:
    """Business-facing churn evaluation output.

    Attributes:
        model_name: Algorithm used (e.g. "XGBoost").
        total_customers: Number of customers scored.
        predicted_churners: Number with predicted churn = 1 (threshold 0.5).
        churn_rate_pct: Percentage of customers predicted to churn.
        high_confidence_churners: Number with P(churn) > 0.7.
        revenue_at_risk_usd: Estimated revenue loss if churners leave.
        top_churn_drivers: Feature importance mapped to business language.
        segment_churn_rates: Churn rate by loyalty_tier (if available).
        recommended_interventions: Prioritised list of retention actions.
    """

    model_name: str
    total_customers: int
    predicted_churners: int
    churn_rate_pct: float
    high_confidence_churners: int
    revenue_at_risk_usd: float
    top_churn_drivers: list[dict[str, object]] = field(default_factory=list)
    segment_churn_rates: dict[str, float] = field(default_factory=dict)
    recommended_interventions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary suitable for a client brief."""
        lines = [
            f"=== Churn Evaluation Report: {self.model_name} ===",
            f"Total customers scored  : {self.total_customers:,}",
            f"Predicted churners      : {self.predicted_churners:,} ({self.churn_rate_pct:.1f}%)",
            f"High-confidence (>70%)  : {self.high_confidence_churners:,}",
            f"Revenue at risk         : ${self.revenue_at_risk_usd:,.0f}",
        ]
        if self.segment_churn_rates:
            lines.append("\nChurn rate by loyalty tier:")
            for tier, rate in sorted(self.segment_churn_rates.items()):
                lines.append(f"  {tier:10s}: {rate:.1%}")
        if self.top_churn_drivers:
            lines.append("\nTop churn drivers:")
            for drv in self.top_churn_drivers[:5]:
                lines.append(f"  {drv['feature']:35s}: {drv['business_meaning']}")
        if self.recommended_interventions:
            lines.append("\nRecommended interventions:")
            for i, action in enumerate(self.recommended_interventions, 1):
                lines.append(f"  {i}. {action}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


# Average annual value per customer used for revenue-at-risk calculation
_AVG_CUSTOMER_ANNUAL_VALUE_USD: float = 850.0

# Threshold for "high confidence" churn flag
_HIGH_CONFIDENCE_THRESHOLD: float = 0.7

# Business meanings for known features
_BUSINESS_MEANINGS: dict[str, str] = {
    "days_since_last_purchase": "Customers who haven't bought recently are at high churn risk",
    "recency_score": "Low recency score indicates the customer hasn't engaged lately",
    "frequency_score": "Infrequent buyers are more likely to churn",
    "monetary_score": "Low-spend customers show higher churn propensity",
    "total_transactions": "Customers with few transactions have weak loyalty",
    "total_spend": "Low lifetime spend signals limited brand attachment",
    "avg_basket_size": "Declining basket size is an early churn signal",
    "discount_sensitivity": "Customers who only buy on discount show much higher churn risk",
    "months_as_customer": "Newer customers churn at higher rates than tenured ones",
    "loyalty_tier": "Bronze-tier customers churn significantly more than Gold/Silver",
    "acquisition_channel": "Channel of acquisition predicts long-term retention",
    "age_group": "Demographic segment influences churn behaviour",
    "preferred_category": "Category preference affects substitution risk",
}


class ChurnEvaluator:
    """Generates business-aware churn evaluation reports.

    Usage::

        evaluator = ChurnEvaluator(avg_customer_value=850.0)
        report = evaluator.evaluate(pipeline, X, y, customers, transactions)
    """

    def __init__(self, avg_customer_value: float = _AVG_CUSTOMER_ANNUAL_VALUE_USD) -> None:
        self._avg_customer_value = avg_customer_value

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> ChurnEvaluationReport:
        """Produce a full business evaluation report for the churn model.

        Args:
            model: Fitted sklearn Pipeline (preprocessor + classifier).
            X: Feature matrix with the same columns used during training.
            y: Ground-truth churn labels (0/1 or bool).
            customers: Customer master table with customer_id, loyalty_tier.
            transactions: Raw transaction data (used for enrichment if needed).

        Returns:
            ChurnEvaluationReport with all fields populated.
        """
        logger.info(f"Evaluating churn model on {len(X):,} customers")

        churn_scores_df = self.generate_churn_scores(model, X)
        proba = churn_scores_df["churn_probability"].values
        y_pred = (proba >= 0.5).astype(int)

        total = len(X)
        n_churners = int(y_pred.sum())
        churn_rate = n_churners / total * 100
        n_high_conf = int((proba >= _HIGH_CONFIDENCE_THRESHOLD).sum())
        revenue_at_risk = n_churners * self._avg_customer_value

        # Feature importance → business drivers
        top_drivers = self._get_top_drivers(model, X)

        # Churn by loyalty tier
        segment_rates = self._compute_segment_churn_rates(
            churn_scores_df, customers, X
        )

        # Recommended interventions
        interventions = self._generate_interventions(
            top_drivers=top_drivers,
            churn_rate_pct=churn_rate,
            n_high_conf=n_high_conf,
            segment_rates=segment_rates,
        )

        model_name = type(model.named_steps["classifier"]).__name__

        report = ChurnEvaluationReport(
            model_name=model_name,
            total_customers=total,
            predicted_churners=n_churners,
            churn_rate_pct=churn_rate,
            high_confidence_churners=n_high_conf,
            revenue_at_risk_usd=revenue_at_risk,
            top_churn_drivers=top_drivers,
            segment_churn_rates=segment_rates,
            recommended_interventions=interventions,
        )
        logger.info(
            f"Churn report: {n_churners:,}/{total:,} predicted churners "
            f"| Revenue at risk: ${revenue_at_risk:,.0f}"
        )
        return report

    def generate_churn_scores(
        self,
        model: Pipeline,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate churn probability scores for every customer.

        Args:
            model: Fitted sklearn Pipeline.
            X: Feature matrix. The index is expected to be customer_id.

        Returns:
            DataFrame with columns [customer_id, churn_probability, churn_flag].
        """
        try:
            proba = model.predict_proba(X)[:, 1]
        except AttributeError:
            proba = model.predict(X).astype(float)

        scores = pd.DataFrame(
            {
                "customer_id": X.index,
                "churn_probability": proba,
                "churn_flag": (proba >= 0.5).astype(int),
            }
        ).reset_index(drop=True)

        return scores

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_top_drivers(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        top_n: int = 10,
    ) -> list[dict[str, object]]:
        """Extract top feature importances and map to business language."""
        clf = model.named_steps["classifier"]
        feature_names = self._get_transformed_feature_names(model, X)

        importances: np.ndarray | None = None
        try:
            importances = clf.feature_importances_
        except AttributeError:
            try:
                importances = np.abs(clf.coef_[0])
            except AttributeError:
                pass

        if importances is None or len(importances) == 0:
            return []

        # Align feature names length with importances
        n = min(len(feature_names), len(importances))
        feature_names = feature_names[:n]
        importances = importances[:n]

        sorted_idx = np.argsort(importances)[::-1][:top_n]
        drivers = []
        for idx in sorted_idx:
            fname = feature_names[idx]
            imp = float(importances[idx])
            # Map raw feature name to base name for business lookup
            base = self._resolve_base_feature(fname)
            meaning = _BUSINESS_MEANINGS.get(base, f"Feature '{fname}' influences churn prediction")
            drivers.append(
                {
                    "feature": fname,
                    "importance": round(imp, 6),
                    "business_meaning": meaning,
                }
            )
        return drivers

    def _get_transformed_feature_names(
        self,
        model: Pipeline,
        X: pd.DataFrame,
    ) -> list[str]:
        """Extract feature names after the preprocessor step."""
        try:
            preprocessor = model.named_steps["preprocessor"]
            names = list(preprocessor.get_feature_names_out())
            return names
        except Exception:
            return list(X.columns)

    def _resolve_base_feature(self, feature_name: str) -> str:
        """Map a potentially encoded feature name back to its base name."""
        # OneHotEncoded features look like "loyalty_tier_Gold"
        for base in _BUSINESS_MEANINGS:
            if feature_name.startswith(base) or feature_name == base:
                return base
        return feature_name

    def _compute_segment_churn_rates(
        self,
        scores: pd.DataFrame,
        customers: pd.DataFrame,
        X: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute predicted churn rate per loyalty tier."""
        segment_rates: dict[str, float] = {}

        if customers is None or customers.empty:
            return segment_rates

        try:
            cust_sub = customers[["customer_id", "loyalty_tier"]].copy()
            merged = scores.merge(cust_sub, on="customer_id", how="left")
            if "loyalty_tier" in merged.columns:
                rates = merged.groupby("loyalty_tier")["churn_flag"].mean()
                segment_rates = rates.to_dict()
        except Exception as exc:
            logger.warning(f"Could not compute segment churn rates: {exc}")

        return segment_rates

    def _generate_interventions(
        self,
        top_drivers: list[dict[str, object]],
        churn_rate_pct: float,
        n_high_conf: int,
        segment_rates: dict[str, float],
    ) -> list[str]:
        """Generate prioritised retention intervention recommendations."""
        interventions: list[str] = []

        # High-confidence churners → immediate action
        if n_high_conf > 0:
            interventions.append(
                f"Immediately target {n_high_conf:,} high-confidence churners (P>70%)"
                " with personalised win-back offers"
            )

        # Churn rate context
        if churn_rate_pct > 20:
            interventions.append(
                "Overall churn rate is elevated — initiate a loyalty programme audit"
            )

        # Feature-specific interventions
        top_feature_names = [d["feature"] for d in top_drivers[:5]]
        for fname in top_feature_names:
            base = self._resolve_base_feature(fname)
            if base == "days_since_last_purchase":
                interventions.append(
                    "Launch re-engagement campaign for customers inactive >60 days"
                )
            elif base == "discount_sensitivity":
                interventions.append(
                    "Reduce discount dependency by introducing value-added loyalty benefits"
                )
            elif base in ("recency_score", "frequency_score"):
                interventions.append(
                    "Introduce a frequency reward multiplier to encourage repeat purchases"
                )
            elif base == "loyalty_tier":
                interventions.append(
                    "Accelerate Bronze→Silver tier progression to reduce Bronze churn"
                )

        # Segment-specific
        if segment_rates:
            worst_tier = max(segment_rates, key=lambda t: segment_rates[t])
            worst_rate = segment_rates[worst_tier]
            if worst_rate > 0.15:
                interventions.append(
                    f"Prioritise {worst_tier} loyalty tier ({worst_rate:.0%} churn rate)"
                    " with targeted retention budget"
                )

        # De-duplicate and cap
        seen: set[str] = set()
        unique_interventions: list[str] = []
        for item in interventions:
            if item not in seen:
                seen.add(item)
                unique_interventions.append(item)

        return unique_interventions[:8]
