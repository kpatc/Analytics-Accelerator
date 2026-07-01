"""Churn prediction pipeline factory.

Builds sklearn Pipelines with full preprocessing for each candidate algorithm.
Answers the business question: "Which customers will stop buying from NovaMart?"

Feature set is fixed at module level so that downstream callers (trainer, evaluator,
SHAP explainer) can reference canonical column lists without importing the trainer.
"""

from __future__ import annotations

import warnings

from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature / target manifest
# ---------------------------------------------------------------------------

CHURN_NUMERIC_FEATURES: list[str] = [
    "days_since_last_purchase",
    "total_transactions",
    "total_spend",
    "avg_basket_size",
    "discount_sensitivity",
    "months_as_customer",
    "recency_score",
    "frequency_score",
    "monetary_score",
]

CHURN_CATEGORICAL_FEATURES: list[str] = [
    "loyalty_tier",
    "acquisition_channel",
    "age_group",
    "preferred_category",
]

ALL_CHURN_FEATURES: list[str] = CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES

CHURN_TARGET: str = "is_churned"


# ---------------------------------------------------------------------------
# Preprocessing sub-pipeline
# ---------------------------------------------------------------------------

def _build_preprocessor() -> ColumnTransformer:
    """Build the column-wise preprocessing transformer.

    Numeric path  : SimpleImputer(median) → StandardScaler
    Categorical path: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, CHURN_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CHURN_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS: dict[str, str] = {
    "LogisticRegression": "LogisticRegression",
    "RandomForest": "RandomForest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
}


def build_churn_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Build a complete sklearn Pipeline for churn prediction.

    The pipeline includes:
    1. ColumnTransformer that applies:
       - Numeric path : SimpleImputer(median) → StandardScaler
       - Categorical path : SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
    2. A classifier step with class_weight='balanced' for all tree models.

    Args:
        model_name: One of "LogisticRegression", "RandomForest", "XGBoost",
                    "LightGBM", "CatBoost".
        random_state: Seed for reproducibility.

    Returns:
        Unfitted sklearn Pipeline.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {list(_SUPPORTED_MODELS)}"
        )

    preprocessor = _build_preprocessor()
    classifier = _build_classifier(model_name, random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    logger.debug(f"Built churn pipeline: {model_name}")
    return pipeline


def _build_classifier(model_name: str, random_state: int) -> object:
    """Instantiate the classifier for a given algorithm name."""

    if model_name == "LogisticRegression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
            C=1.0,
        )

    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # handled via compute_sample_weight or eval_metric
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )

    if model_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                verbosity=-1,
                force_row_wise=True,
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception as exc:
            logger.warning(f"LightGBM unavailable ({exc}), falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )

    if model_name == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=random_state,
        )

    raise ValueError(f"Unhandled model: {model_name}")
