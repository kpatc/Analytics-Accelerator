"""MLflow-tracked price elasticity model training orchestrator.

Fits the ElasticityModelPipeline on NovaMart transaction + product data,
logs per-category elasticities and model R² to MLflow.

MLflow experiment: "bcgx-novamart-elasticity"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from bcgx.models.base import ModelMetrics, TrainingResult
from bcgx.models.price_elasticity.pipeline import ElasticityModelPipeline


class ElasticityTrainer:
    """Orchestrates price elasticity model training with MLflow tracking.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI.
    """

    def __init__(
        self,
        experiment_name: str = "bcgx-novamart-elasticity",
        tracking_uri: str = "mlruns",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._model: ElasticityModelPipeline | None = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(
            f"ElasticityTrainer initialised — experiment='{experiment_name}'"
            f" tracking_uri='{tracking_uri}'"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        transactions: pd.DataFrame,
        products: pd.DataFrame,
        alpha: float = 0.01,
    ) -> TrainingResult:
        """Fit the elasticity model and log results to MLflow.

        Args:
            transactions: Transaction data (product_id, unit_price, quantity, etc.).
            products: Product catalogue (product_id, category, brand_type, unit_cost).
            alpha: Ridge regularisation strength.

        Returns:
            TrainingResult with per-category elasticity metrics.
        """
        logger.info(
            f"Starting elasticity training: {len(transactions):,} transactions"
            f" | {products['product_id'].nunique():,} products"
            f" | {products['category'].nunique()} categories"
        )

        pipeline = ElasticityModelPipeline(alpha=alpha, min_obs_per_product=5)

        parent_run_id: str | None = None
        experiment_id: str | None = None
        all_metrics: dict[str, ModelMetrics] = {}

        with mlflow.start_run(run_name="elasticity_training") as parent_run:
            parent_run_id = parent_run.info.run_id
            experiment_id = parent_run.info.experiment_id

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("n_transactions", len(transactions))
            mlflow.log_param("n_products", products["product_id"].nunique())
            mlflow.log_param("n_categories", products["category"].nunique())

            # Fit model
            pipeline.fit(transactions, products)
            self._model = pipeline

            # Get all elasticities
            elasticity_df = pipeline.get_all_elasticities()
            n_products_fitted = len(elasticity_df)
            mlflow.log_metric("n_products_fitted", n_products_fitted)

            # Per-category metrics
            if not elasticity_df.empty and "category" in elasticity_df.columns:
                cat_stats = elasticity_df.groupby("category").agg(
                    mean_elasticity=("elasticity", "mean"),
                    mean_r2=("r_squared", "mean"),
                    n_products=("product_id", "count"),
                )

                for cat, row in cat_stats.iterrows():
                    with mlflow.start_run(run_name=f"elasticity_{cat}", nested=True):
                        mlflow.log_metric("mean_elasticity", float(row["mean_elasticity"]))
                        mlflow.log_metric("mean_r2", float(row["mean_r2"]))
                        mlflow.log_metric("n_products", int(row["n_products"]))
                        mlflow.log_param("category", cat)

                    all_metrics[str(cat)] = ModelMetrics(
                        model_name=f"Elasticity_{cat}",
                        model_type="regression",
                        r_squared=float(row["mean_r2"]),
                    )

                # Log aggregate stats
                global_mean_elasticity = float(elasticity_df["elasticity"].mean())
                global_mean_r2 = float(elasticity_df["r_squared"].mean())
                mlflow.log_metric("global_mean_elasticity", global_mean_elasticity)
                mlflow.log_metric("global_mean_r2", global_mean_r2)

                # Log category elasticities as JSON artifact
                cat_elast_dict = cat_stats["mean_elasticity"].to_dict()
                cat_path = "/tmp/category_elasticities.json"
                with open(cat_path, "w") as f:
                    json.dump(
                        {str(k): float(v) for k, v in cat_elast_dict.items()},
                        f,
                        indent=2,
                    )
                mlflow.log_artifact(cat_path, "elasticity_analysis")

                # Save full elasticity table
                elast_path = "/tmp/product_elasticities.parquet"
                elasticity_df.to_parquet(elast_path, index=False)
                mlflow.log_artifact(elast_path, "elasticity_analysis")

                logger.info(
                    f"Global mean elasticity: {global_mean_elasticity:.3f}"
                    f" | Global mean R²: {global_mean_r2:.3f}"
                )

                best_cat = cat_stats["mean_r2"].idxmax()
                best_metrics = ModelMetrics(
                    model_name="Elasticity_global",
                    model_type="regression",
                    r_squared=global_mean_r2,
                    business_metric_name="Global mean price elasticity",
                    business_metric_value=global_mean_elasticity,
                    business_metric_description=(
                        f"Average price elasticity across {n_products_fitted:,} products"
                        f" and {len(cat_stats)} categories"
                    ),
                )
            else:
                best_metrics = ModelMetrics(
                    model_name="Elasticity_global",
                    model_type="regression",
                )

        logger.success("Elasticity training complete")
        return TrainingResult(
            model_name="price_elasticity",
            best_model_name="ElasticityModel_Ridge",
            metrics=all_metrics,
            best_metrics=best_metrics,
            feature_names=["log_price", "brand_code", "log_price_x_brand_code"],
            mlflow_run_id=parent_run_id,
            mlflow_experiment_id=experiment_id,
            model_path=None,
        )

    def save_model(self, output_dir: str = "data/outputs/models") -> str:
        """Serialize the fitted elasticity model to disk.

        Args:
            output_dir: Directory to write the model file.

        Returns:
            Path to the saved model file.
        """
        if self._model is None:
            raise RuntimeError("No model trained yet. Call train() first.")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_file = out_path / "price_elasticity_model.joblib"
        joblib.dump(self._model, model_file)
        logger.success(f"Elasticity model saved → {model_file}")
        return str(model_file)
