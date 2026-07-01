"""MLflow-tracked store performance model training orchestrator.

Trains four candidate regressors (Ridge, RandomForest, XGBoost, LightGBM) on
the NovaMart store feature matrix to predict avg_gross_margin. Also trains a
classification head to identify is_top_performer.

Business metric: "Revenue uplift potential" — for the bottom 20% of stores,
the gap to the median performance × number of underperforming stores × avg monthly revenue.

MLflow experiment: "bcgx-novamart-store-performance"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from bcgx.models.base import ModelMetrics, TrainingResult
from bcgx.models.store_performance.pipeline import (
    ALL_STORE_FEATURES,
    STORE_CATEGORICAL_FEATURES,
    STORE_NUMERIC_FEATURES,
    STORE_REGRESSION_TARGET,
    STORE_CLASSIFICATION_TARGET,
    build_store_pipeline,
)

_CANDIDATE_MODELS: list[str] = ["Ridge", "RandomForest", "XGBoost", "LightGBM"]


class StorePerformanceTrainer:
    """Orchestrates store performance model training with MLflow experiment tracking.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI (defaults to local "mlruns/" directory).
        task: "regression" (default) or "classification".
    """

    def __init__(
        self,
        experiment_name: str = "bcgx-novamart-store-performance",
        tracking_uri: str = "mlruns",
        task: str = "regression",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._task = task
        self._best_pipeline: Pipeline | None = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(
            f"StorePerformanceTrainer initialised — task='{task}'"
            f" experiment='{experiment_name}'"
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
        """Train all candidate store performance models.

        Args:
            features: Store feature matrix from StoreFeatureEngineer.build().
                      Must contain ALL_STORE_FEATURES columns and the target column.
            test_size: Fraction held out for test evaluation.
            n_cv_folds: Number of folds for cross-validation.
            random_state: Reproducibility seed.

        Returns:
            TrainingResult with best model, all metrics, and MLflow run IDs.
        """
        target_col = (
            STORE_REGRESSION_TARGET
            if self._task == "regression"
            else STORE_CLASSIFICATION_TARGET
        )
        logger.info(
            f"Starting store performance training on {len(features):,} stores"
            f" | task={self._task} | target={target_col}"
        )

        # Ensure categorical columns exist
        for col in STORE_CATEGORICAL_FEATURES:
            if col not in features.columns:
                features = features.copy()
                features[col] = "unknown"

        feature_cols = [c for c in ALL_STORE_FEATURES if c in features.columns]
        X = features[feature_cols].copy()

        if target_col not in features.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in features. "
                f"Available: {list(features.columns)}"
            )
        y = features[target_col]

        # ---- Train / test split ----
        if self._task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y.astype(int), random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        all_metrics: dict[str, ModelMetrics] = {}
        best_pipeline: Pipeline | None = None
        best_score: float = -np.inf
        best_model_name: str = ""
        parent_run_id: str | None = None
        experiment_id: str | None = None

        with mlflow.start_run(
            run_name=f"store_{self._task}_seed{random_state}"
        ) as parent_run:
            parent_run_id = parent_run.info.run_id
            experiment_id = parent_run.info.experiment_id
            mlflow.log_param("task", self._task)
            mlflow.log_param("target", target_col)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_cv_folds", n_cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", len(feature_cols))

            for model_name in _CANDIDATE_MODELS:
                try:
                    pipeline = build_store_pipeline(
                        model_name, task=self._task, random_state=random_state
                    )
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

                    # Primary score for model selection
                    primary = (
                        metrics.auc_roc
                        if self._task == "classification"
                        else metrics.r_squared
                    )
                    if primary is not None:
                        logger.info(
                            f"  {model_name:20s} primary={primary:.4f}"
                            + (
                                f" AUC={metrics.auc_roc:.4f}"
                                if self._task == "classification"
                                else f" RMSE={metrics.rmse:.4f} R²={metrics.r_squared:.4f}"
                            )
                        )
                        if primary > best_score:
                            best_score = primary
                            best_pipeline = pipeline
                            best_model_name = model_name

                except Exception as exc:
                    logger.warning(f"Model {model_name} failed: {exc}")
                    continue

            if not all_metrics:
                raise RuntimeError("All candidate store performance models failed.")

            best_metrics = all_metrics[best_model_name]

            # ---- Business metric: Revenue uplift potential ----
            if best_pipeline is not None and self._task == "regression":
                y_pred_all = best_pipeline.predict(X)
                bottom_20_threshold = np.percentile(y_pred_all, 20)
                median_margin = float(np.median(y_pred_all))
                bottom_stores = y_pred_all < bottom_20_threshold
                n_bottom = int(bottom_stores.sum())
                avg_rev = float(
                    features.get("avg_monthly_revenue", pd.Series([100_000])).mean()
                )
                # Uplift = (median_margin - avg_bottom_margin) * avg_rev * n_bottom
                avg_bottom_margin = float(y_pred_all[bottom_stores].mean()) if n_bottom > 0 else 0.0
                uplift_usd = max(0.0, (median_margin - avg_bottom_margin)) * avg_rev * n_bottom
                best_metrics.business_metric_name = "Revenue uplift potential (USD)"
                best_metrics.business_metric_value = uplift_usd
                best_metrics.business_metric_description = (
                    f"{n_bottom} underperforming stores × "
                    f"${avg_rev:,.0f} avg monthly revenue × margin gap to median"
                )
                mlflow.log_metric("revenue_uplift_potential_usd", uplift_usd)
                mlflow.log_metric("n_underperforming_stores", n_bottom)

            mlflow.log_metric("best_primary_score", best_score)
            mlflow.log_param("best_model", best_model_name)

            comparison = {
                name: {k: v for k, v in m.to_dict().items() if isinstance(v, (int, float))}
                for name, m in all_metrics.items()
            }
            comparison_path = "/tmp/store_model_comparison.json"
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)
            mlflow.log_artifact(comparison_path, "model_comparison")

        self._best_pipeline = best_pipeline
        logger.success(
            f"Best store model: {best_model_name} (score={best_score:.4f})"
        )

        return TrainingResult(
            model_name="store_performance",
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
        """Fit one pipeline, evaluate on test set, run CV, log to MLflow child run."""
        with mlflow.start_run(run_name=f"store_{model_name}", nested=True):
            pipeline.fit(X_train, y_train)

            # Log hyperparameters
            step_key = "regressor" if self._task == "regression" else "classifier"
            try:
                est = pipeline.named_steps[step_key]
                params = est.get_params()
                loggable = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
                mlflow.log_params({f"{model_name}__{k}": v for k, v in loggable.items()})
            except Exception:
                pass

            y_pred = pipeline.predict(X_test)

            if self._task == "regression":
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))

                # MAPE — guard against zeros
                mask = y_test != 0
                mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100) \
                    if mask.sum() > 0 else 0.0

                cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
                )
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())

                mlflow.log_metrics(
                    {
                        "rmse": rmse,
                        "mae": mae,
                        "r_squared": r2,
                        "mape": mape,
                        "cv_mean_r2": cv_mean,
                        "cv_std_r2": cv_std,
                    }
                )
                mlflow.log_param("model_name", model_name)

                return ModelMetrics(
                    model_name=model_name,
                    model_type="regression",
                    rmse=rmse,
                    mae=mae,
                    r_squared=r2,
                    mape=mape,
                    cv_mean=cv_mean,
                    cv_std=cv_std,
                )

            else:
                # Classification
                try:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                except AttributeError:
                    y_proba = y_pred.astype(float)

                auc = float(roc_auc_score(y_test, y_proba))
                ap = float(average_precision_score(y_test, y_proba))
                f1 = float(f1_score(y_test, y_pred, zero_division=0))
                prec = float(precision_score(y_test, y_pred, zero_division=0))
                rec = float(recall_score(y_test, y_pred, zero_division=0))

                cv = StratifiedKFold(
                    n_splits=n_cv_folds, shuffle=True, random_state=random_state
                )
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train.astype(int), cv=cv, scoring="roc_auc", n_jobs=-1
                )
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())

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
        """Serialize the best pipeline to disk.

        Args:
            result: TrainingResult returned by train_all_models().
            output_dir: Directory to save the model file.

        Returns:
            Absolute path of the saved model file.
        """
        if self._best_pipeline is None:
            raise RuntimeError("No model trained yet. Call train_all_models() first.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fname = f"store__{self._task}__{result.best_model_name.lower()}.joblib"
        model_file = out_path / fname
        joblib.dump(self._best_pipeline, model_file)

        meta = {
            "model_name": result.model_name,
            "best_model_name": result.best_model_name,
            "task": self._task,
            "feature_names": result.feature_names,
            "mlflow_run_id": result.mlflow_run_id,
        }
        meta_file = out_path / fname.replace(".joblib", "_meta.json")
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.success(f"Best store model saved → {model_file}")
        result.model_path = str(model_file)
        return str(model_file)
