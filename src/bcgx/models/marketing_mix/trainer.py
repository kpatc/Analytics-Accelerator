"""MLflow-tracked Marketing Mix Model training orchestrator.

Fits MMM separately for urban vs rural store formats to capture ROI differences
by location — a key analytical insight for NovaMart's media planning.

MLflow experiment: "bcgx-novamart-mmm"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score

from bcgx.models.base import ModelMetrics, TrainingResult
from bcgx.models.marketing_mix.pipeline import DEFAULT_DECAY_RATES, MMMPipeline

# Store format groupings
_URBAN_FORMATS: set[str] = {"urban", "city_centre", "high_street", "city"}
_RURAL_FORMATS: set[str] = {"rural", "out_of_town", "suburban", "retail_park"}

# Fallback monthly budget allocation (sum = 1.0)
_DEFAULT_BUDGET_SPLIT: dict[str, float] = {
    "tv": 0.30,
    "digital": 0.25,
    "email": 0.05,
    "radio": 0.15,
    "outdoor": 0.15,
    "social": 0.10,
}


class MMMTrainer:
    """Orchestrates Marketing Mix Model training with MLflow tracking.

    Trains separate models per store_format segment (urban / rural / all)
    to surface location-specific ROI differences.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI.
    """

    def __init__(
        self,
        experiment_name: str = "bcgx-novamart-mmm",
        tracking_uri: str = "mlruns",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._models: dict[str, MMMPipeline] = {}
        self._channel_names: list[str] = []

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(
            f"MMMTrainer initialised — experiment='{experiment_name}'"
            f" tracking_uri='{tracking_uri}'"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        marketing: pd.DataFrame,
        transactions: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> TrainingResult:
        """Train MMM models by store segment.

        Args:
            marketing: Monthly marketing spend with columns:
                       store_id, year_month, channel, spend_usd.
            transactions: Transaction data with store_id, year_month, gross_revenue.
            stores: Store master with store_id, store_format.

        Returns:
            TrainingResult with per-segment metrics and MLflow run IDs.
        """
        logger.info("Starting MMM training")

        # ---- Build monthly revenue per store ----
        monthly_rev = (
            transactions.groupby(["store_id", "year_month"])["gross_revenue"]
            .sum()
            .reset_index(name="revenue")
        )

        # ---- Build pivot: year_month × channel spend (aggregated across stores) ----
        # Then segment by store format
        store_format = stores[["store_id", "store_format"]].copy()
        store_format["format_group"] = store_format["store_format"].apply(
            self._classify_format
        )

        monthly_rev = monthly_rev.merge(store_format, on="store_id", how="left")
        marketing_merged = marketing.merge(store_format, on="store_id", how="left")

        all_metrics: dict[str, ModelMetrics] = {}
        parent_run_id: str | None = None
        experiment_id: str | None = None

        with mlflow.start_run(run_name="mmm_full_training") as parent_run:
            parent_run_id = parent_run.info.run_id
            experiment_id = parent_run.info.experiment_id

            # Train per segment
            for segment in ["urban", "rural", "all"]:
                try:
                    metrics = self._train_segment(
                        segment=segment,
                        marketing_merged=marketing_merged,
                        monthly_rev=monthly_rev,
                    )
                    all_metrics[segment] = metrics
                    logger.info(
                        f"  Segment '{segment}': R²={metrics.r_squared:.4f}"
                        + (f" RMSE={metrics.rmse:.4f}" if metrics.rmse else "")
                    )
                except Exception as exc:
                    logger.warning(f"MMM segment '{segment}' failed: {exc}")

            # Log channel ROIs for the "all" segment
            if "all" in self._models:
                rois = self._models["all"].get_roi_by_channel()
                contributions = self._models["all"].get_channel_contributions()
                for ch, roi in rois.items():
                    mlflow.log_metric(f"roi_all_{ch}", roi)
                for ch, contrib in contributions.items():
                    mlflow.log_metric(f"contrib_all_{ch}_pct", contrib)

                # Urban vs rural ROI comparison
                if "urban" in self._models and "rural" in self._models:
                    urban_rois = self._models["urban"].get_roi_by_channel()
                    rural_rois = self._models["rural"].get_roi_by_channel()
                    roi_comparison = {
                        ch: {
                            "urban": urban_rois.get(ch, 0.0),
                            "rural": rural_rois.get(ch, 0.0),
                        }
                        for ch in self._channel_names
                    }
                    roi_path = "/tmp/mmm_roi_comparison.json"
                    with open(roi_path, "w") as f:
                        json.dump(roi_comparison, f, indent=2)
                    mlflow.log_artifact(roi_path, "mmm_analysis")

        best_metrics = all_metrics.get("all", list(all_metrics.values())[0]) if all_metrics else ModelMetrics(
            model_name="MMM", model_type="regression"
        )
        best_r2 = best_metrics.r_squared or 0.0

        logger.success(f"MMM training complete. Global R²={best_r2:.4f}")
        return TrainingResult(
            model_name="mmm",
            best_model_name="RidgeMMM",
            metrics=all_metrics,
            best_metrics=best_metrics,
            feature_names=self._channel_names,
            mlflow_run_id=parent_run_id,
            mlflow_experiment_id=experiment_id,
            model_path=None,
        )

    def get_budget_recommendation(
        self,
        total_budget: float,
        store_format: str = "all",
    ) -> dict[str, float]:
        """Recommend channel budget allocation to maximise revenue.

        Allocates budget proportional to channel ROI for the given store format segment.

        Args:
            total_budget: Total marketing budget in USD.
            store_format: One of "urban", "rural", "all".

        Returns:
            Dict mapping channel_name → recommended spend in USD.
        """
        segment = self._classify_format(store_format)
        if segment not in self._models:
            segment = "all"

        if segment not in self._models:
            logger.warning("No trained MMM model found — using default budget split")
            return {ch: total_budget * share for ch, share in _DEFAULT_BUDGET_SPLIT.items()}

        rois = self._models[segment].get_roi_by_channel()
        channels = [ch for ch in rois if rois[ch] > 0]

        if not channels:
            return {ch: total_budget * share for ch, share in _DEFAULT_BUDGET_SPLIT.items()}

        # Allocate proportionally to ROI
        total_roi = sum(rois[ch] for ch in channels)
        allocation = {
            ch: round(rois[ch] / total_roi * total_budget, 2) for ch in channels
        }

        logger.info(
            f"Budget recommendation for '{store_format}': "
            + " | ".join(f"{ch}=${v:,.0f}" for ch, v in sorted(allocation.items()))
        )
        return allocation

    def save_models(self, output_dir: str = "data/outputs/models") -> str:
        """Serialize all trained MMM models to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for segment, model in self._models.items():
            model_file = out_path / f"mmm_{segment}.joblib"
            joblib.dump(model, model_file)
        logger.success(f"MMM models saved to {output_dir}")
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _train_segment(
        self,
        segment: str,
        marketing_merged: pd.DataFrame,
        monthly_rev: pd.DataFrame,
    ) -> ModelMetrics:
        """Train one MMM model for a given store format segment."""
        with mlflow.start_run(run_name=f"mmm_{segment}", nested=True):
            # Filter by segment
            if segment != "all":
                mkt_seg = marketing_merged[marketing_merged["format_group"] == segment]
                rev_seg = monthly_rev[monthly_rev["format_group"] == segment]
            else:
                mkt_seg = marketing_merged
                rev_seg = monthly_rev

            if len(rev_seg) < 4:
                raise ValueError(
                    f"Not enough data for segment '{segment}': {len(rev_seg)} rows"
                )

            # Aggregate spend per year_month × channel
            spend_pivot = (
                mkt_seg.groupby(["year_month", "channel"])["spend_usd"]
                .sum()
                .unstack(fill_value=0.0)
                .sort_index()
            )
            self._channel_names = list(spend_pivot.columns)

            # Aggregate revenue per year_month
            rev_agg = rev_seg.groupby("year_month")["revenue"].sum().sort_index()

            # Align on common months
            common_months = spend_pivot.index.intersection(rev_agg.index)
            if len(common_months) < 4:
                raise ValueError(
                    f"Only {len(common_months)} overlapping months for segment '{segment}'"
                )
            spend_aligned = spend_pivot.loc[common_months]
            rev_aligned = rev_agg.loc[common_months]

            # Fit MMM
            mmm = MMMPipeline(alpha=1.0)
            mmm.fit(spend_aligned, rev_aligned)
            self._models[segment] = mmm

            # Evaluate in-sample
            y_pred = mmm.predict(spend_aligned)
            y_true = rev_aligned.values
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred))
            mae = float(np.mean(np.abs(y_true - y_pred)))

            # ROI and contributions
            rois = mmm.get_roi_by_channel()
            contributions = mmm.get_channel_contributions()

            # Log to MLflow
            mlflow.log_param("segment", segment)
            mlflow.log_param("n_months", len(common_months))
            mlflow.log_param("n_channels", len(self._channel_names))
            mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
            for ch, roi in rois.items():
                mlflow.log_metric(f"roi_{ch}", roi)
            for ch, contrib in contributions.items():
                mlflow.log_metric(f"contrib_{ch}_pct", contrib)

            return ModelMetrics(
                model_name=f"MMM_{segment}",
                model_type="regression",
                rmse=rmse,
                mae=mae,
                r_squared=r2,
            )

    def _classify_format(self, store_format: str) -> str:
        """Map a store format to urban / rural / all segment."""
        fmt_lower = str(store_format).lower()
        if any(u in fmt_lower for u in _URBAN_FORMATS):
            return "urban"
        if any(r in fmt_lower for r in _RURAL_FORMATS):
            return "rural"
        return "all"
