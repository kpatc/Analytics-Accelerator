"""SHAP-based model explanations for NovaMart ML models.

Produces interpretable, consulting-ready explanations of model predictions
by computing global SHAP feature importance and translating feature names
into business language.

Supports:
- Tree-based models (XGBoost, LightGBM, RandomForest) via shap.TreeExplainer
- Linear models (LogisticRegression, Ridge) via shap.LinearExplainer
- Handles sklearn Pipeline objects by extracting the final estimator and
  transforming X through the preprocessor step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class SHAPExplanation:
    """Container for SHAP-based model explanation.

    Attributes:
        model_name: Algorithm name (e.g. "XGBoostClassifier").
        feature_names: Feature names after preprocessing.
        mean_abs_shap: Mean |SHAP| per feature.
        top_features: Top features with business meaning annotations.
        global_explanation: Natural-language summary of the model's key drivers.
        shap_values: Raw SHAP values array (n_samples, n_features). May be None
                     if computation was skipped.
    """

    model_name: str
    feature_names: list[str]
    mean_abs_shap: dict[str, float]
    top_features: list[dict[str, object]]
    global_explanation: str
    shap_values: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Business meaning lookup
# ---------------------------------------------------------------------------

# Mapping from base feature name → (positive SHAP direction, business explanation)
_BUSINESS_MEANINGS: dict[str, dict[str, str]] = {
    # Churn features
    "days_since_last_purchase": {
        "high_positive": "Customers who haven't bought recently are much more likely to churn",
        "high_negative": "Recent buyers are at low churn risk",
        "default": "Recency of last purchase is a key churn predictor",
    },
    "discount_sensitivity": {
        "high_positive": "Customers who only buy on discount show much higher churn risk",
        "high_negative": "Price-insensitive customers are more loyal",
        "default": "Discount sensitivity indicates churn-prone behaviour",
    },
    "recency_score": {
        "high_positive": "High recency score indicates recent engagement — low churn risk",
        "high_negative": "Low recency score signals disengagement — high churn risk",
        "default": "Recency score is a strong predictor of retention",
    },
    "frequency_score": {
        "high_positive": "Frequent shoppers are more likely to stay",
        "high_negative": "Infrequent buyers are at elevated churn risk",
        "default": "Purchase frequency drives loyalty and retention",
    },
    "monetary_score": {
        "high_positive": "High-value customers are more likely to stay",
        "high_negative": "Low-spend customers show higher churn propensity",
        "default": "Customer lifetime value is predictive of retention",
    },
    "total_transactions": {
        "high_positive": "Customers with many transactions have strong brand attachment",
        "high_negative": "Customers with few transactions have weak loyalty",
        "default": "Transaction history depth predicts churn likelihood",
    },
    "total_spend": {
        "high_positive": "High lifetime spend signals deep brand engagement",
        "high_negative": "Low lifetime spend indicates weak brand attachment",
        "default": "Lifetime spend is a proxy for customer commitment",
    },
    "avg_basket_size": {
        "high_positive": "Large baskets signal committed shoppers",
        "high_negative": "Declining basket size is an early churn signal",
        "default": "Average basket size reflects shopping engagement",
    },
    "months_as_customer": {
        "high_positive": "Tenured customers are significantly less likely to churn",
        "high_negative": "Newer customers are at higher churn risk",
        "default": "Customer tenure strongly predicts retention",
    },
    "loyalty_tier": {
        "high_positive": "Higher tier (Gold/Silver) customers show strong retention",
        "high_negative": "Bronze-tier customers churn significantly more",
        "default": "Loyalty tier is a strong predictor of retention behaviour",
    },
    # Store performance features
    "manager_tenure_years": {
        "high_positive": "Store manager experience significantly improves performance",
        "high_negative": "New store managers correlate with underperformance",
        "default": "Manager tenure is a key driver of store success",
    },
    "avg_monthly_revenue": {
        "high_positive": "Higher-revenue stores generate disproportionate profit",
        "high_negative": "Low-revenue stores face structural margin challenges",
        "default": "Monthly revenue scale is fundamental to profitability",
    },
    "operating_cost_ratio": {
        "high_positive": "Low cost ratio is the strongest driver of margin improvement",
        "high_negative": "High operating costs severely constrain profitability",
        "default": "Cost efficiency is the primary margin lever",
    },
    "marketing_efficiency": {
        "high_positive": "High marketing ROI drives sustained revenue growth",
        "high_negative": "Low marketing efficiency signals budget misallocation",
        "default": "Marketing efficiency predicts revenue performance",
    },
    "digital_spend_share": {
        "high_positive": "Digital-first marketing delivers higher efficiency",
        "high_negative": "Low digital adoption limits marketing effectiveness",
        "default": "Digital marketing channel mix affects store growth",
    },
    "revenue_trend_slope": {
        "high_positive": "Positive revenue trend indicates a growing store",
        "high_negative": "Declining revenue trend is a leading underperformance indicator",
        "default": "Revenue trajectory strongly predicts future performance",
    },
    "revenue_volatility": {
        "high_positive": "High volatility can indicate seasonal opportunity",
        "high_negative": "Volatile revenue makes consistent profitability harder",
        "default": "Revenue stability is a key performance predictor",
    },
    "sq_footage": {
        "high_positive": "Larger stores capture more footfall and product range",
        "high_negative": "Small format stores face space constraints on profitability",
        "default": "Store size shapes revenue capacity and customer experience",
    },
}


def _resolve_base_feature(feature_name: str) -> str:
    """Map a potentially encoded feature name to its base name."""
    for base in _BUSINESS_MEANINGS:
        if feature_name == base or feature_name.startswith(f"{base}_"):
            return base
    return feature_name


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

_TREE_CLASSES = (
    "XGBClassifier",
    "XGBRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
)

_LINEAR_CLASSES = (
    "LogisticRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LinearRegression",
    "SGDClassifier",
    "SGDRegressor",
)


class SHAPExplainer:
    """Generates SHAP-based explanations for any sklearn Pipeline model.

    Handles Pipeline objects by:
    1. Transforming X through the preprocessor to get the model's input space.
    2. Using TreeExplainer for tree-based models, LinearExplainer for linear models.
    3. Falling back to KernelExplainer for any other model type.

    Usage::

        explainer = SHAPExplainer()
        explanation = explainer.explain_classification_model(pipeline, X_test, "XGBoost")
        print(explanation.global_explanation)
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def explain_classification_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        model_name: str,
        n_samples: int = 500,
    ) -> SHAPExplanation:
        """Compute SHAP explanations for a classification Pipeline.

        Args:
            model: Fitted sklearn Pipeline (preprocessor + classifier).
            X: Feature DataFrame (same columns used at training time).
            model_name: Human-readable model name for the explanation.
            n_samples: Number of samples to use for SHAP (random subsample if larger).

        Returns:
            SHAPExplanation with global feature importance and business meanings.
        """
        return self._explain(model, X, model_name, task="classification", n_samples=n_samples)

    def explain_regression_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        model_name: str,
        n_samples: int = 500,
    ) -> SHAPExplanation:
        """Compute SHAP explanations for a regression Pipeline.

        Args:
            model: Fitted sklearn Pipeline (preprocessor + regressor).
            X: Feature DataFrame.
            model_name: Human-readable model name.
            n_samples: Sample limit for SHAP computation.

        Returns:
            SHAPExplanation with global feature importance and business meanings.
        """
        return self._explain(model, X, model_name, task="regression", n_samples=n_samples)

    def get_business_meaning(self, feature_name: str, shap_value: float) -> str:
        """Translate a feature name and its SHAP value into consulting language.

        Args:
            feature_name: Raw or encoded feature name.
            shap_value: Mean SHAP value (positive = increases prediction).

        Returns:
            Business-friendly explanation string.
        """
        base = _resolve_base_feature(feature_name)
        meanings = _BUSINESS_MEANINGS.get(base, {})
        if not meanings:
            direction = "increases" if shap_value > 0 else "decreases"
            return f"Feature '{feature_name}' {direction} the model prediction"

        if shap_value > 0.05:
            return meanings.get("high_positive", meanings.get("default", ""))
        elif shap_value < -0.05:
            return meanings.get("high_negative", meanings.get("default", ""))
        else:
            return meanings.get("default", "")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _explain(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        model_name: str,
        task: str,
        n_samples: int,
    ) -> SHAPExplanation:
        """Core explanation logic."""
        # Subsample for speed
        if len(X) > n_samples:
            rng = np.random.default_rng(self._seed)
            idx = rng.choice(len(X), size=n_samples, replace=False)
            X_sample = X.iloc[idx]
        else:
            X_sample = X

        # Extract preprocessor and estimator from Pipeline
        estimator, X_transformed, feature_names = self._extract_from_pipeline(
            model, X_sample
        )

        # Determine explainer type
        est_class = type(estimator).__name__
        shap_values = self._compute_shap(estimator, X_transformed, est_class, task)

        if shap_values is None or shap_values.size == 0:
            logger.warning(f"SHAP computation returned empty values for {model_name}")
            return SHAPExplanation(
                model_name=model_name,
                feature_names=feature_names,
                mean_abs_shap={},
                top_features=[],
                global_explanation=f"Could not compute SHAP explanations for {model_name}",
                shap_values=None,
            )

        # Mean absolute SHAP per feature
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        n = min(len(feature_names), len(mean_abs))
        feature_names = feature_names[:n]
        mean_abs = mean_abs[:n]

        mean_abs_dict = {feature_names[i]: float(mean_abs[i]) for i in range(n)}

        # Top features
        sorted_idx = np.argsort(mean_abs)[::-1][:15]
        top_features = []
        for idx in sorted_idx:
            fname = feature_names[idx]
            shap_val = float(mean_abs[idx])
            meaning = self.get_business_meaning(fname, shap_val)
            top_features.append(
                {
                    "feature": fname,
                    "shap_value": round(shap_val, 6),
                    "business_meaning": meaning,
                }
            )

        # Build global explanation string
        top3 = [d["feature"] for d in top_features[:3]]
        task_label = "churn" if task == "classification" else "performance"
        global_explanation = (
            f"The top drivers of {task_label} for model '{model_name}' are: "
            + ", ".join(top3[:3])
            + ". "
            + (top_features[0]["business_meaning"] if top_features else "")
        )

        logger.info(
            f"SHAP explanation: top features = {', '.join(top3)}"
        )

        return SHAPExplanation(
            model_name=model_name,
            feature_names=feature_names,
            mean_abs_shap=mean_abs_dict,
            top_features=top_features,
            global_explanation=global_explanation,
            shap_values=shap_values,
        )

    def _extract_from_pipeline(
        self,
        model: Pipeline,
        X: pd.DataFrame,
    ) -> tuple[object, np.ndarray, list[str]]:
        """Extract the estimator and transformed X from a sklearn Pipeline.

        Returns:
            (estimator, X_transformed, feature_names)
        """
        # Try to find preprocessor and estimator by name
        estimator = None
        preprocessor = None

        for name, step in model.steps:
            if name == "preprocessor":
                preprocessor = step
            elif name in ("classifier", "regressor"):
                estimator = step
            else:
                estimator = step  # last step

        if estimator is None:
            estimator = model[-1]

        # Transform X
        if preprocessor is not None:
            try:
                X_transformed = preprocessor.transform(X)
                try:
                    feature_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    n_cols = X_transformed.shape[1] if hasattr(X_transformed, "shape") else len(X.columns)
                    feature_names = [f"feature_{i}" for i in range(n_cols)]
            except Exception as exc:
                logger.debug(f"Preprocessor transform failed: {exc}")
                X_transformed = X.values.astype(float)
                feature_names = list(X.columns)
        else:
            X_transformed = X.values.astype(float)
            feature_names = list(X.columns)

        # Ensure numpy array
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_transformed = np.asarray(X_transformed, dtype=float)

        return estimator, X_transformed, feature_names

    def _compute_shap(
        self,
        estimator: object,
        X_transformed: np.ndarray,
        est_class: str,
        task: str,
    ) -> np.ndarray | None:
        """Compute SHAP values using the appropriate explainer."""
        try:
            if est_class in _TREE_CLASSES:
                return self._shap_tree(estimator, X_transformed, task)
            elif est_class in _LINEAR_CLASSES:
                return self._shap_linear(estimator, X_transformed)
            else:
                return self._shap_kernel(estimator, X_transformed, task)
        except Exception as exc:
            logger.warning(f"SHAP computation failed ({est_class}): {exc}")
            return None

    def _shap_tree(
        self,
        estimator: object,
        X: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """Compute SHAP values using TreeExplainer."""
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(X, check_additivity=False)

        # For binary classification, TreeExplainer may return list [class0, class1]
        if isinstance(raw, list):
            if len(raw) == 2:
                return np.array(raw[1])  # class 1 (churn / top performer)
            return np.array(raw[0])

        return np.array(raw)

    def _shap_linear(
        self,
        estimator: object,
        X: np.ndarray,
    ) -> np.ndarray:
        """Compute SHAP values using LinearExplainer."""
        background = shap.maskers.Independent(X, max_samples=min(100, len(X)))
        explainer = shap.LinearExplainer(estimator, background)
        raw = explainer.shap_values(X)
        if isinstance(raw, list):
            return np.array(raw[-1])
        return np.array(raw)

    def _shap_kernel(
        self,
        estimator: object,
        X: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """Compute SHAP values using KernelExplainer (model-agnostic, slower)."""
        background = shap.sample(X, min(50, len(X)), random_state=self._seed)

        if task == "classification" and hasattr(estimator, "predict_proba"):
            predict_fn = lambda x: estimator.predict_proba(x)[:, 1]  # noqa: E731
        else:
            predict_fn = estimator.predict  # type: ignore[assignment]

        explainer = shap.KernelExplainer(predict_fn, background)
        n_eval = min(100, len(X))
        raw = explainer.shap_values(X[:n_eval], nsamples=100)
        if isinstance(raw, list):
            raw = raw[-1]
        return np.array(raw)
