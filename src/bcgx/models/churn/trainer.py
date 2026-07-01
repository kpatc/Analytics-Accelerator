"""MLflow-tracked churn model training orchestrator.

Trains five candidate classifiers (LogisticRegression, RandomForest, XGBoost,
LightGBM, CatBoost) on the NovaMart customer feature matrix, logs every
experiment to MLflow, and returns a TrainingResult with the best model selected
by AUC-ROC on the held-out test split.

Business metric logged: "Revenue at risk" = n_predicted_churners × avg_customer_annual_value.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from bcgx.models.base import ModelMetrics, TrainingResult
from bcgx.models.churn.pipeline import (
    ALL_CHURN_FEATURES,
    CHURN_CATEGORICAL_FEATURES,
    CHURN_NUMERIC_FEATURES,
    CHURN_TARGET,
    build_churn_pipeline,
)

# Average customer annual value — used for the business metric
_AVG_CUSTOMER_ANNUAL_VALUE_USD: float = 850.0

_CANDIDATE_MODELS: list[str] = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


class ChurnModelTrainer:
    """Orchestrates churn model training with MLflow experiment tracking.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI (defaults to local "mlruns/" directory).
    """

    def __init__(
        self,
        experiment_name: str = "bcgx-novamart-churn",
        tracking_uri: str = "mlruns",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._best_pipeline: Pipeline | None = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(
            f"ChurnModelTrainer initialised — experiment='{experiment_name}'"
            f" tracking_uri='{tracking_uri}'"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train_all_models(
        self,
        features: pd.DataFrame,
        test_size: float = 0.2,
        n_cv_folds: int = 5,
        random_state: int = 42,
    ) -> TrainingResult:
        """Train all candidate churn models and return a TrainingResult.

        Args:
            features: Customer feature matrix produced by CustomerFeatureEngineer.build().
                      Must contain ALL_CHURN_FEATURES columns and the CHURN_TARGET column.
            test_size: Fraction of data held out for test evaluation.
            n_cv_folds: Number of StratifiedKFold folds for cross-validation.
            random_state: Random seed for all stochastic operations.

        Returns:
            TrainingResult with best model, all metrics, and MLflow run IDs.
        """
        logger.info(
            f"Starting churn model training on {len(features):,} customers"
            f" (test_size={test_size}, n_cv_folds={n_cv_folds})"
        )

        # ---- Prepare feature matrix ----
        available_numeric = [c for c in CHURN_NUMERIC_FEATURES if c in features.columns]
        available_categorical = [c for c in CHURN_CATEGORICAL_FEATURES if c in features.columns]
        feature_cols = available_numeric + available_categorical

        # For categorical features missing from the engineer output, fill with "unknown"
        for col in CHURN_CATEGORICAL_FEATURES:
            if col not in features.columns:
                features = features.copy()
                features[col] = "unknown"

        X = features[feature_cols].copy()
        y = features[CHURN_TARGET].astype(int)

        # ---- Train / test split ----
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        logger.info(
            f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
            f"Churn rate: {y.mean():.1%}"
        )

        # ---- Train each candidate ----
        all_metrics: dict[str, ModelMetrics] = {}
        best_pipeline: Pipeline | None = None
        best_auc: float = -1.0
        best_model_name: str = ""
        parent_run_id: str | None = None
        experiment_id: str | None = None

        with mlflow.start_run(run_name=f"churn_training_seed{random_state}") as parent_run:
            parent_run_id = parent_run.info.run_id
            experiment_id = parent_run.info.experiment_id
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_cv_folds", n_cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("churn_rate", float(y.mean()))
            mlflow.log_param("n_features", len(feature_cols))

            for model_name in _CANDIDATE_MODELS:
                try:
                    pipeline = build_churn_pipeline(model_name, random_state=random_state)
                    metrics = self._train_single_model(
                        model_name=model_name,
                        pipeline=pipeline,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        n_cv_folds=n_cv_folds,
                        random_state=random_state,
                    )
                    all_metrics[model_name] = metrics
                    logger.info(
                        f"  {model_name:25s} AUC={metrics.auc_roc:.4f}"
                        f"  F1={metrics.f1_score:.4f}"
                    )

                    if metrics.auc_roc is not None and metrics.auc_roc > best_auc:
                        best_auc = metrics.auc_roc
                        best_pipeline = pipeline
                        best_model_name = model_name

                except Exception as exc:
                    logger.warning(f"Model {model_name} failed: {exc}")
                    continue

            if not all_metrics:
                raise RuntimeError("All candidate churn models failed to train.")

            best_metrics = all_metrics[best_model_name]

            # ---- Business metric: revenue at risk ----
            if best_pipeline is not None:
                y_pred = best_pipeline.predict(X_test)
                n_predicted_churners = int(y_pred.sum())
                revenue_at_risk = n_predicted_churners * _AVG_CUSTOMER_ANNUAL_VALUE_USD
                best_metrics.business_metric_name = "Revenue at risk (USD)"
                best_metrics.business_metric_value = revenue_at_risk
                best_metrics.business_metric_description = (
                    f"{n_predicted_churners:,} predicted churners"
                    f" × ${_AVG_CUSTOMER_ANNUAL_VALUE_USD:,.0f} avg annual value"
                )
                mlflow.log_metric("best_auc_roc", best_auc)
                mlflow.log_metric("revenue_at_risk_usd", revenue_at_risk)
                mlflow.log_metric("n_predicted_churners", n_predicted_churners)
                mlflow.log_param("best_model", best_model_name)

                # Log model comparison as JSON artifact
                comparison = {
                    name: {k: v for k, v in m.to_dict().items() if isinstance(v, (int, float))}
                    for name, m in all_metrics.items()
                }
                comparison_path = "/tmp/churn_model_comparison.json"
                with open(comparison_path, "w") as f:
                    json.dump(comparison, f, indent=2)
                mlflow.log_artifact(comparison_path, "model_comparison")

        self._best_pipeline = best_pipeline
        logger.success(
            f"Best churn model: {best_model_name} (AUC={best_auc:.4f})"
        )

        return TrainingResult(
            model_name="churn",
            best_model_name=best_model_name,
            metrics=all_metrics,
            best_metrics=best_metrics,
            feature_names=feature_cols,
            mlflow_run_id=parent_run_id,
            mlflow_experiment_id=experiment_id,
            model_path=None,
        )

    def _train_single_model(
        self,
        model_name: str,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        n_cv_folds: int = 5,
        random_state: int = 42,
    ) -> ModelMetrics:
        """Train one pipeline, evaluate on test set, run CV, log to MLflow child run.

        Args:
            model_name: Human-readable algorithm name.
            pipeline: Unfitted sklearn Pipeline (preprocessor + classifier).
            X_train / y_train: Training data.
            X_test / y_test: Hold-out test data.
            n_cv_folds: Number of StratifiedKFold folds.
            random_state: Seed for CV.

        Returns:
            ModelMetrics populated with test and CV metrics.
        """
        with mlflow.start_run(run_name=model_name, nested=True) as child_run:
            # Fit
            pipeline.fit(X_train, y_train)

            # Log classifier hyperparameters
            clf = pipeline.named_steps["classifier"]
            try:
                params = clf.get_params()
                loggable = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
                mlflow.log_params({f"{model_name}__{k}": v for k, v in loggable.items()})
            except Exception:
                pass

            # Test set predictions
            y_pred = pipeline.predict(X_test)
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_proba = y_pred.astype(float)

            # Compute metrics
            auc = float(roc_auc_score(y_test, y_proba))
            ap = float(average_precision_score(y_test, y_proba))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))

            # 5-fold StratifiedKFold CV on training data
            cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "auc_roc": auc,
                    "avg_precision": ap,
                    "f1_score": f1,
                    "precision": prec,
                    "recall": rec,
                    "cv_mean_auc": cv_mean,
                    "cv_std_auc": cv_std,
                }
            )
            mlflow.log_param("model_name", model_name)

            # Log confusion matrix data as JSON artifact
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_test, y_pred).tolist()
            cm_path = f"/tmp/churn_cm_{model_name}.json"
            with open(cm_path, "w") as f:
                json.dump({"confusion_matrix": cm, "model": model_name}, f)
            mlflow.log_artifact(cm_path, "confusion_matrices")

        return ModelMetrics(
            model_name=model_name,
            model_type="classification",
            auc_roc=auc,
            avg_precision=ap,
            f1_score=f1,
            precision=prec,
            recall=rec,
            cv_mean=cv_mean,
            cv_std=cv_std,
        )

    def save_best_model(
        self,
        result: TrainingResult,
        output_dir: str = "data/outputs/models",
    ) -> str:
        """Serialize the best pipeline to disk using joblib.

        Args:
            result: TrainingResult returned by train_all_models().
            output_dir: Directory to save the model file.

        Returns:
            Absolute path of the saved model file.

        Raises:
            RuntimeError: If no model has been trained yet.
        """
        if self._best_pipeline is None:
            raise RuntimeError(
                "No model has been trained yet. Call train_all_models() first."
            )
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / f"churn__{result.best_model_name.lower()}.joblib"
        joblib.dump(self._best_pipeline, model_file)

        # Also save metadata alongside the model
        meta = {
            "model_name": result.model_name,
            "best_model_name": result.best_model_name,
            "feature_names": result.feature_names,
            "mlflow_run_id": result.mlflow_run_id,
            "auc_roc": result.best_metrics.auc_roc,
            "f1_score": result.best_metrics.f1_score,
        }
        meta_file = out_path / f"churn__{result.best_model_name.lower()}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.success(f"Best churn model saved → {model_file}")
        result.model_path = str(model_file)
        return str(model_file)
