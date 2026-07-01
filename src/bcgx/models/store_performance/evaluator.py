"""Business-aware store performance evaluator.

Translates raw model predictions into consulting-ready insights:
- Revenue uplift potential for underperforming stores
- Stores to invest in vs. monitor
- Key performance drivers with business context
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
class StorePerformanceReport:
    """Business-facing store performance evaluation output.

    Attributes:
        model_name: Algorithm used.
        total_stores: Number of stores scored.
        top_performers: Stores predicted as top performers.
        underperformers: Stores predicted below median.
        revenue_uplift_potential_usd: If underperformers reach median performance.
        key_performance_drivers: Feature importance mapped to business language.
        stores_to_invest: Store IDs with highest uplift potential.
        stores_to_monitor: Store IDs predicted to decline.
    """

    model_name: str
    total_stores: int
    top_performers: int
    underperformers: int
    revenue_uplift_potential_usd: float
    key_performance_drivers: list[dict[str, object]] = field(default_factory=list)
    stores_to_invest: list[str] = field(default_factory=list)
    stores_to_monitor: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary for client reporting."""
        lines = [
            f"=== Store Performance Report: {self.model_name} ===",
            f"Total stores scored   : {self.total_stores}",
            f"Top performers        : {self.top_performers} ({self.top_performers / max(self.total_stores, 1):.0%})",
            f"Underperformers       : {self.underperformers} ({self.underperformers / max(self.total_stores, 1):.0%})",
            f"Revenue uplift pot.   : ${self.revenue_uplift_potential_usd:,.0f}",
        ]
        if self.stores_to_invest:
            lines.append(
                f"\nTop stores to invest  : {', '.join(self.stores_to_invest[:10])}"
            )
        if self.stores_to_monitor:
            lines.append(
                f"Stores to monitor     : {', '.join(self.stores_to_monitor[:10])}"
            )
        if self.key_performance_drivers:
            lines.append("\nKey performance drivers:")
            for drv in self.key_performance_drivers[:5]:
                lines.append(f"  {drv['feature']:35s}: {drv['business_meaning']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Business meaning lookup
# ---------------------------------------------------------------------------

_STORE_BUSINESS_MEANINGS: dict[str, str] = {
    "avg_monthly_revenue": "Higher-revenue stores generate disproportionate profit",
    "sq_footage": "Larger stores have more product range and footfall capacity",
    "manager_tenure_years": "Store manager experience significantly improves operational performance",
    "marketing_efficiency": "ROI on marketing spend is a leading indicator of top performance",
    "digital_spend_share": "Digital marketing channels deliver higher efficiency in urban formats",
    "operating_cost_ratio": "Lower cost ratio is the strongest driver of margin improvement",
    "revenue_trend_slope": "Stores with positive revenue trends are on a path to top performance",
    "revenue_volatility": "High volatility stores struggle to build consistent profitability",
    "months_since_open": "Newer stores are still building their customer base and supply chain",
    "store_format": "Store format (urban/suburban/rural) fundamentally shapes performance range",
    "region": "Regional dynamics significantly affect competitive intensity and costs",
    "performance_cluster": "Historical cluster classification predicts future performance trajectory",
}


class StorePerformanceEvaluator:
    """Generates business-aware store performance evaluation reports.

    Usage::

        evaluator = StorePerformanceEvaluator()
        report = evaluator.evaluate(pipeline, X, features, task="regression")
    """

    def __init__(self, avg_monthly_revenue: float = 150_000.0) -> None:
        """
        Args:
            avg_monthly_revenue: Used as fallback when feature data doesn't include revenues.
        """
        self._avg_monthly_revenue = avg_monthly_revenue

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        features: pd.DataFrame,
        task: str = "regression",
    ) -> StorePerformanceReport:
        """Generate a full business evaluation report for the store performance model.

        Args:
            model: Fitted sklearn Pipeline.
            X: Feature matrix with the same columns used during training.
            features: Full store feature matrix (may include additional columns for business metrics).
            task: "regression" or "classification".

        Returns:
            StorePerformanceReport with all fields populated.
        """
        logger.info(f"Evaluating store performance model on {len(X):,} stores (task={task})")

        y_pred = model.predict(X)
        store_ids = list(X.index.astype(str))
        total = len(X)

        # Get avg revenue if available
        if "avg_monthly_revenue" in features.columns:
            avg_rev_series = features["avg_monthly_revenue"]
        else:
            avg_rev_series = pd.Series(
                [self._avg_monthly_revenue] * len(features), index=features.index
            )

        if task == "regression":
            median_margin = float(np.median(y_pred))
            bottom_20_threshold = float(np.percentile(y_pred, 20))
            top_20_threshold = float(np.percentile(y_pred, 80))

            is_top = y_pred >= top_20_threshold
            is_bottom = y_pred < bottom_20_threshold

            n_top = int(is_top.sum())
            n_bottom = int(is_bottom.sum())

            # Revenue uplift: if bottom stores reached median
            bottom_indices = np.where(is_bottom)[0]
            avg_bottom_margin = float(y_pred[bottom_indices].mean()) if n_bottom > 0 else 0.0
            avg_rev_bottom = float(avg_rev_series.iloc[bottom_indices].mean()) if n_bottom > 0 else self._avg_monthly_revenue
            gap = max(0.0, median_margin - avg_bottom_margin)
            uplift_usd = gap * avg_rev_bottom * n_bottom * 12  # annualised

            # Identify stores to invest (bottom-20 with highest revenue = highest uplift)
            if n_bottom > 0:
                bottom_store_data = pd.DataFrame(
                    {
                        "store_id": [store_ids[i] for i in bottom_indices],
                        "pred_margin": y_pred[bottom_indices],
                        "avg_rev": avg_rev_series.iloc[bottom_indices].values,
                    }
                )
                bottom_store_data["uplift_potential"] = (
                    (median_margin - bottom_store_data["pred_margin"])
                    * bottom_store_data["avg_rev"]
                )
                top_invest = bottom_store_data.nlargest(20, "uplift_potential")
                stores_to_invest = list(top_invest["store_id"])
            else:
                stores_to_invest = []

            # Stores to monitor: those near bottom but not yet there
            monitor_threshold = float(np.percentile(y_pred, 35))
            monitor_indices = np.where(
                (y_pred < monitor_threshold) & (y_pred >= bottom_20_threshold)
            )[0]
            stores_to_monitor = [store_ids[i] for i in monitor_indices[:20]]

        else:
            # Classification
            try:
                y_proba = model.predict_proba(X)[:, 1]
            except AttributeError:
                y_proba = y_pred.astype(float)

            n_top = int(y_pred.sum())
            n_bottom = total - n_top

            is_bottom = y_pred == 0
            bottom_indices = np.where(is_bottom)[0]

            # Estimate uplift from going from predicted 0 → predicted 1
            uplift_usd = n_bottom * self._avg_monthly_revenue * 0.05 * 12

            low_conf_top = (y_proba >= 0.4) & (y_proba < 0.5)
            stores_to_invest = [store_ids[i] for i in np.where(low_conf_top)[0][:20]]
            high_risk = y_proba < 0.3
            stores_to_monitor = [store_ids[i] for i in np.where(high_risk)[0][:20]]

        # Feature importance
        drivers = self._get_key_drivers(model, X)
        model_name = type(
            model.named_steps.get("regressor", model.named_steps.get("classifier", model[-1]))
        ).__name__

        report = StorePerformanceReport(
            model_name=model_name,
            total_stores=total,
            top_performers=n_top,
            underperformers=n_bottom,
            revenue_uplift_potential_usd=uplift_usd,
            key_performance_drivers=drivers,
            stores_to_invest=stores_to_invest,
            stores_to_monitor=stores_to_monitor,
        )
        logger.info(
            f"Store report: {n_top}/{total} top performers | "
            f"Uplift potential: ${uplift_usd:,.0f}"
        )
        return report

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_key_drivers(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        top_n: int = 10,
    ) -> list[dict[str, object]]:
        """Extract feature importances and map to business language."""
        # Get step name (regressor or classifier)
        step_key = None
        for name in ("regressor", "classifier"):
            if name in model.named_steps:
                step_key = name
                break

        if step_key is None:
            return []

        est = model.named_steps[step_key]
        try:
            feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
        except Exception:
            feature_names = list(X.columns)

        importances: np.ndarray | None = None
        try:
            importances = est.feature_importances_
        except AttributeError:
            try:
                importances = np.abs(est.coef_.ravel())
            except AttributeError:
                pass

        if importances is None:
            return []

        n = min(len(feature_names), len(importances))
        feature_names = feature_names[:n]
        importances = importances[:n]

        sorted_idx = np.argsort(importances)[::-1][:top_n]
        drivers = []
        for idx in sorted_idx:
            fname = feature_names[idx]
            imp = float(importances[idx])
            base = self._resolve_base(fname)
            meaning = _STORE_BUSINESS_MEANINGS.get(
                base, f"Feature '{fname}' drives store performance prediction"
            )
            drivers.append(
                {
                    "feature": fname,
                    "importance": round(imp, 6),
                    "business_meaning": meaning,
                }
            )
        return drivers

    def _resolve_base(self, feature_name: str) -> str:
        """Map encoded feature name to base feature for business lookup."""
        for base in _STORE_BUSINESS_MEANINGS:
            if feature_name.startswith(base) or feature_name == base:
                return base
        return feature_name
