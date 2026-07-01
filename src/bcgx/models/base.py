"""Abstract base class and shared data contracts for all NovaMart ML models.

Provides:
- ModelMetrics: typed container for classification and regression evaluation results.
- TrainingResult: typed container for a full training run (multiple algorithms compared).
- BaseModel: ABC that all concrete model classes must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics.

    Supports both classification and regression metrics. Fields not applicable
    to the model type are left as None.
    """

    model_name: str
    model_type: str  # "classification" | "regression"

    # ---- Classification ----
    auc_roc: float | None = None
    avg_precision: float | None = None
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None

    # ---- Regression ----
    rmse: float | None = None
    mae: float | None = None
    r_squared: float | None = None
    mape: float | None = None

    # ---- Cross-validation ----
    cv_mean: float | None = None
    cv_std: float | None = None

    # ---- Business metric ----
    business_metric_name: str | None = None
    business_metric_value: float | None = None
    business_metric_description: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return all non-None metrics as a flat dict (suitable for MLflow logging)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class TrainingResult:
    """Container for the output of a full model selection training run.

    One TrainingResult is returned per model type (churn, store, etc.) after all
    candidate algorithms have been trained and compared.
    """

    model_name: str  # logical name, e.g. "churn"
    best_model_name: str  # winning algorithm, e.g. "XGBoost"
    metrics: dict[str, ModelMetrics]  # algorithm_name -> metrics
    best_metrics: ModelMetrics
    feature_names: list[str]
    mlflow_run_id: str | None
    mlflow_experiment_id: str | None
    model_path: str | None

    def summary(self) -> str:
        """Return a human-readable summary of the training result."""
        lines = [
            f"Model: {self.model_name}",
            f"Best algorithm: {self.best_model_name}",
        ]
        m = self.best_metrics
        if m.auc_roc is not None:
            lines.append(f"  AUC-ROC : {m.auc_roc:.4f}")
        if m.f1_score is not None:
            lines.append(f"  F1      : {m.f1_score:.4f}")
        if m.rmse is not None:
            lines.append(f"  RMSE    : {m.rmse:.4f}")
        if m.r_squared is not None:
            lines.append(f"  R²      : {m.r_squared:.4f}")
        if m.cv_mean is not None:
            lines.append(f"  CV mean : {m.cv_mean:.4f} ± {m.cv_std:.4f}")
        if m.business_metric_name:
            lines.append(
                f"  {m.business_metric_name}: ${m.business_metric_value:,.0f}"
                if m.business_metric_value is not None
                else f"  {m.business_metric_name}: N/A"
            )
        if self.mlflow_run_id:
            lines.append(f"  MLflow run: {self.mlflow_run_id}")
        return "\n".join(lines)


class BaseModel(ABC):
    """Abstract base class for all NovaMart ML model wrappers.

    Concrete subclasses must implement train/predict/predict_proba/get_feature_importance.
    Each subclass is expected to wrap one or more sklearn Pipelines internally and
    expose the winning pipeline via `self.best_pipeline`.
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model on training data.

        Args:
            X_train: Feature matrix (rows = samples, columns = features).
            y_train: Target series aligned with X_train.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard predictions.

        Args:
            X: Feature matrix.

        Returns:
            1-D array of predictions (class labels or regression values).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates (classification only).

        Args:
            X: Feature matrix.

        Returns:
            2-D array of shape (n_samples, n_classes).
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending.

        Returns:
            Series indexed by feature name, values = importance score.
        """
        ...
