"""Store performance prediction pipeline factory.

Builds sklearn Pipelines for:
- Regression: predicting avg_gross_margin (continuous)
- Classification: predicting is_top_performer (binary)

Answers the business question: "Which stores should receive investment?"

Five candidate algorithms are supported:
1. Ridge (baseline regression) / LogisticRegression (baseline classification)
2. RandomForestRegressor / RandomForestClassifier
3. XGBRegressor / XGBClassifier
4. LGBMRegressor / LGBMClassifier (with fallback)
5. GradientBoostingRegressor / GradientBoostingClassifier (sklearn fallback)
"""

from __future__ import annotations

from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature / target manifest
# ---------------------------------------------------------------------------

STORE_NUMERIC_FEATURES: list[str] = [
    "avg_monthly_revenue",
    "sq_footage",
    "manager_tenure_years",
    "marketing_efficiency",
    "digital_spend_share",
    "operating_cost_ratio",
    "revenue_trend_slope",
    "revenue_volatility",
    "months_since_open",
]

STORE_CATEGORICAL_FEATURES: list[str] = [
    "store_format",
    "region",
    "performance_cluster",
]

ALL_STORE_FEATURES: list[str] = STORE_NUMERIC_FEATURES + STORE_CATEGORICAL_FEATURES

STORE_REGRESSION_TARGET: str = "avg_gross_margin"
STORE_CLASSIFICATION_TARGET: str = "is_top_performer"

_SUPPORTED_REGRESSION_MODELS: dict[str, str] = {
    "Ridge": "Ridge",
    "RandomForest": "RandomForest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}

_SUPPORTED_CLASSIFICATION_MODELS: dict[str, str] = {
    "LogisticRegression": "LogisticRegression",
    "RandomForest": "RandomForest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _build_store_preprocessor() -> ColumnTransformer:
    """Build the column-wise preprocessing transformer for store features.

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
            ("numeric", numeric_pipeline, STORE_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, STORE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------


def build_store_pipeline(
    model_name: str,
    task: str = "regression",
    random_state: int = 42,
) -> Pipeline:
    """Build a complete sklearn Pipeline for store performance prediction.

    Args:
        model_name: One of "Ridge", "RandomForest", "XGBoost", "LightGBM".
        task: "regression" (predicts avg_gross_margin) or "classification"
              (predicts is_top_performer).
        random_state: Seed for reproducibility.

    Returns:
        Unfitted sklearn Pipeline ready for fit/predict.

    Raises:
        ValueError: If model_name or task is not recognised.
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

    preprocessor = _build_store_preprocessor()
    estimator = _build_store_estimator(model_name, task, random_state)

    step_name = "regressor" if task == "regression" else "classifier"
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (step_name, estimator),
        ]
    )
    logger.debug(f"Built store {task} pipeline: {model_name}")
    return pipeline


def _build_store_estimator(model_name: str, task: str, random_state: int) -> object:
    """Instantiate the appropriate estimator."""

    is_regression = task == "regression"

    if model_name == "Ridge":
        if is_regression:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0)
        else:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
                C=1.0,
            )

    if model_name == "RandomForest":
        if is_regression:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )

    if model_name == "XGBoost":
        if is_regression:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
            )
        else:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
            )

    if model_name == "LightGBM":
        try:
            if is_regression:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbosity=-1,
                    force_row_wise=True,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
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
            logger.warning(f"LightGBM unavailable ({exc}), using GradientBoosting fallback")
            if is_regression:
                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=random_state,
                )
            else:
                from sklearn.ensemble import GradientBoostingClassifier

                return GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=random_state,
                )

    raise ValueError(
        f"Unknown model '{model_name}'. Supported: {list(_SUPPORTED_REGRESSION_MODELS)}"
    )
