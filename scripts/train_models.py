"""Train all NovaMart ML models and log experiments to MLflow.

Full training pipeline:
1. Load raw data
2. Build features (CustomerFeatureEngineer, StoreFeatureEngineer, ProductFeatureEngineer)
3. Save to FeatureStore
4. Train selected models (churn / store / mmm / elasticity)
5. Run SHAP explanation on the best model
6. Print results table (rich)
7. Save model comparison JSON to data/outputs/model_comparison.json

Usage:
    python scripts/train_models.py [OPTIONS]
    python scripts/train_models.py --models churn
    python scripts/train_models.py --models churn,store --seed 0 --verbose
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Train all NovaMart ML models.")

# Ensure the src/ directory is on the path when running as a script
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    models: str = typer.Option(
        "all",
        help="Comma-separated list of models to train: churn,store,mmm,elasticity (or 'all')",
    ),
    seed: int = typer.Option(42, help="Random seed for all estimators"),
    tracking_uri: str = typer.Option(
        "mlruns", help="MLflow tracking URI (default: local mlruns/ directory)"
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
    output_dir: str = typer.Option(
        "data/outputs/models", help="Directory to save trained model artefacts"
    ),
    features_dir: str = typer.Option(
        "data/features", help="FeatureStore directory"
    ),
) -> None:
    """Train the full NovaMart ML model suite.

    Each model is:
    1. Trained with cross-validation
    2. Evaluated on a held-out test set
    3. Logged to MLflow (local mlruns/ by default)
    4. Serialised to OUTPUT_DIR as a joblib file

    A model_comparison.json is saved to data/outputs/model_comparison.json.
    """
    # ---- Configure logging ----
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level} | {message}")

    # ---- Resolve which models to train ----
    requested = {m.strip().lower() for m in models.split(",")} if models != "all" else {"all"}
    train_churn = "all" in requested or "churn" in requested
    train_store = "all" in requested or "store" in requested
    train_mmm = "all" in requested or "mmm" in requested
    train_elasticity = "all" in requested or "elasticity" in requested

    console.print(
        f"\n[bold cyan]BCG X NovaMart — ML Model Training[/bold cyan]\n"
        f"  Models   : {models}\n"
        f"  Seed     : {seed}\n"
        f"  MLflow   : {tracking_uri}\n"
        f"  Output   : {output_dir}\n"
    )

    # ---- Load data ----
    console.print("[bold]Step 1/5:[/bold] Loading raw data...")
    from bcgx.data.loader import DataLoader

    loader = DataLoader()
    try:
        data = loader.load_all()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Data not found:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ---- Build features ----
    console.print("[bold]Step 2/5:[/bold] Engineering features...")

    from bcgx.features.customer_features import CustomerFeatureEngineer
    from bcgx.features.feature_store import FeatureStore
    from bcgx.features.product_features import ProductFeatureEngineer
    from bcgx.features.store_features import StoreFeatureEngineer

    fs = FeatureStore(features_dir)

    # Customer features (always built — needed for churn)
    cust_eng = CustomerFeatureEngineer()
    customer_features = cust_eng.build(data["transactions"], data["customers"])
    fs.save_customer_features(customer_features)
    console.print(
        f"  [green]✓[/green] Customer features: {customer_features.shape[0]:,} customers"
        f" × {customer_features.shape[1]} features"
    )

    # Store features (needed for store + mmm)
    store_eng = StoreFeatureEngineer()
    store_features = store_eng.build(
        data["transactions"],
        data["stores"],
        data["marketing_spend"],
        data["store_costs"],
    )
    fs.save_store_features(store_features)
    console.print(
        f"  [green]✓[/green] Store features: {store_features.shape[0]} stores"
        f" × {store_features.shape[1]} features"
    )

    # Product features (needed for elasticity)
    prod_eng = ProductFeatureEngineer()
    product_features = prod_eng.build(data["transactions"], data["products"])
    fs.save_product_features(product_features)
    console.print(
        f"  [green]✓[/green] Product features: {product_features.shape[0]:,} products"
        f" × {product_features.shape[1]} features"
    )

    # ---- Train models ----
    console.print("[bold]Step 3/5:[/bold] Training models...")

    results: dict[str, object] = {}
    comparison: dict[str, dict[str, object]] = {}

    if train_churn:
        console.print("  [yellow]→[/yellow] Training churn models...")
        try:
            from bcgx.models.churn.trainer import ChurnModelTrainer

            trainer = ChurnModelTrainer(tracking_uri=tracking_uri)
            result = trainer.train_all_models(
                features=customer_features,
                test_size=0.2,
                n_cv_folds=5,
                random_state=seed,
            )
            trainer.save_best_model(result, output_dir=output_dir)
            results["churn"] = result
            comparison["churn"] = {
                "best_model": result.best_model_name,
                "auc_roc": result.best_metrics.auc_roc,
                "f1_score": result.best_metrics.f1_score,
                "cv_mean": result.best_metrics.cv_mean,
                "cv_std": result.best_metrics.cv_std,
                "business_metric": result.best_metrics.business_metric_name,
                "business_value": result.best_metrics.business_metric_value,
            }
            console.print(
                f"  [green]✓[/green] Churn: best={result.best_model_name}"
                f" AUC={result.best_metrics.auc_roc:.4f}"
            )
        except Exception as exc:
            logger.error(f"Churn training failed: {exc}")
            console.print(f"  [red]✗[/red] Churn training failed: {exc}")

    if train_store:
        console.print("  [yellow]→[/yellow] Training store performance models...")
        try:
            from bcgx.models.store_performance.trainer import StorePerformanceTrainer

            # Add missing features that store trainer expects
            store_feats = store_features.copy()
            for col in ["marketing_efficiency", "store_format", "region", "performance_cluster"]:
                if col not in store_feats.columns:
                    if col in data["stores"].columns:
                        store_feats = store_feats.merge(
                            data["stores"].set_index("store_id")[[col]],
                            left_index=True,
                            right_index=True,
                            how="left",
                        )
                    else:
                        store_feats[col] = 0.0 if col == "marketing_efficiency" else "unknown"

            # Compute marketing_efficiency if missing
            if "marketing_efficiency" not in store_feats.columns and "total_marketing_spend" in store_feats.columns:
                store_feats["marketing_efficiency"] = (
                    store_feats["avg_monthly_revenue"] / store_feats["total_marketing_spend"].clip(lower=1)
                )

            store_trainer = StorePerformanceTrainer(tracking_uri=tracking_uri)
            store_result = store_trainer.train_all_models(
                features=store_feats,
                test_size=0.2,
                n_cv_folds=5,
                random_state=seed,
            )
            store_trainer.save_best_model(store_result, output_dir=output_dir)
            results["store"] = store_result
            comparison["store"] = {
                "best_model": store_result.best_model_name,
                "r_squared": store_result.best_metrics.r_squared,
                "rmse": store_result.best_metrics.rmse,
                "cv_mean": store_result.best_metrics.cv_mean,
                "business_metric": store_result.best_metrics.business_metric_name,
                "business_value": store_result.best_metrics.business_metric_value,
            }
            console.print(
                f"  [green]✓[/green] Store: best={store_result.best_model_name}"
                f" R²={store_result.best_metrics.r_squared:.4f}"
            )
        except Exception as exc:
            logger.error(f"Store training failed: {exc}")
            console.print(f"  [red]✗[/red] Store training failed: {exc}")

    if train_mmm:
        console.print("  [yellow]→[/yellow] Training Marketing Mix Model...")
        try:
            from bcgx.models.marketing_mix.trainer import MMMTrainer

            mmm_trainer = MMMTrainer(tracking_uri=tracking_uri)
            mmm_result = mmm_trainer.train(
                marketing=data["marketing_spend"],
                transactions=data["transactions"],
                stores=data["stores"],
            )
            mmm_trainer.save_models(output_dir=output_dir)
            results["mmm"] = mmm_result
            comparison["mmm"] = {
                "best_model": mmm_result.best_model_name,
                "r_squared": mmm_result.best_metrics.r_squared,
                "rmse": mmm_result.best_metrics.rmse,
            }
            console.print(
                f"  [green]✓[/green] MMM: R²={mmm_result.best_metrics.r_squared:.4f}"
            )
        except Exception as exc:
            logger.error(f"MMM training failed: {exc}")
            console.print(f"  [red]✗[/red] MMM training failed: {exc}")

    if train_elasticity:
        console.print("  [yellow]→[/yellow] Training Price Elasticity Model...")
        try:
            from bcgx.models.price_elasticity.trainer import ElasticityTrainer

            elast_trainer = ElasticityTrainer(tracking_uri=tracking_uri)
            elast_result = elast_trainer.train(
                transactions=data["transactions"],
                products=data["products"],
            )
            elast_trainer.save_model(output_dir=output_dir)
            results["elasticity"] = elast_result
            comparison["elasticity"] = {
                "best_model": elast_result.best_model_name,
                "r_squared": elast_result.best_metrics.r_squared,
                "mean_elasticity": elast_result.best_metrics.business_metric_value,
            }
            console.print(
                f"  [green]✓[/green] Elasticity:"
                f" R²={elast_result.best_metrics.r_squared:.4f}"
            )
        except Exception as exc:
            logger.error(f"Elasticity training failed: {exc}")
            console.print(f"  [red]✗[/red] Elasticity training failed: {exc}")

    # ---- SHAP explanation on best churn model ----
    console.print("[bold]Step 4/5:[/bold] Running SHAP explanations...")
    if "churn" in results:
        try:
            from bcgx.explainability.shap_explainer import SHAPExplainer
            from bcgx.models.churn.pipeline import ALL_CHURN_FEATURES, CHURN_CATEGORICAL_FEATURES

            churn_result = results["churn"]  # type: ignore[index]
            best_model_path = (
                Path(output_dir) / f"churn__{churn_result.best_model_name.lower()}.joblib"  # type: ignore[union-attr]
            )

            if best_model_path.exists():
                import joblib

                best_pipeline = joblib.load(best_model_path)
                # Use a sample of customer features for SHAP
                X_for_shap = customer_features.copy()
                for col in CHURN_CATEGORICAL_FEATURES:
                    if col not in X_for_shap.columns:
                        X_for_shap[col] = "unknown"

                feat_cols = [c for c in ALL_CHURN_FEATURES if c in X_for_shap.columns]
                X_shap = X_for_shap[feat_cols]

                shap_explainer = SHAPExplainer(seed=seed)
                explanation = shap_explainer.explain_classification_model(
                    best_pipeline, X_shap, churn_result.best_model_name, n_samples=300  # type: ignore[union-attr]
                )
                console.print(
                    f"  [green]✓[/green] SHAP: top features = "
                    + ", ".join(d["feature"] for d in explanation.top_features[:3])
                )
                console.print(f"\n  [italic]{explanation.global_explanation}[/italic]\n")
            else:
                console.print("  [yellow]⚠[/yellow] Best model file not found, skipping SHAP")
        except Exception as exc:
            logger.warning(f"SHAP explanation failed: {exc}")
            console.print(f"  [yellow]⚠[/yellow] SHAP failed (non-fatal): {exc}")

    # ---- Results table ----
    console.print("[bold]Step 5/5:[/bold] Saving results...")

    table = Table(title="Model Training Results", show_header=True, header_style="bold blue")
    table.add_column("Model", style="cyan")
    table.add_column("Best Algorithm", style="yellow")
    table.add_column("Primary Metric", justify="right")
    table.add_column("CV Mean", justify="right")
    table.add_column("Business Metric", justify="right", style="green")

    for model_key, info in comparison.items():
        primary = info.get("auc_roc") or info.get("r_squared")
        primary_str = f"{primary:.4f}" if isinstance(primary, float) else "N/A"
        cv_mean = info.get("cv_mean")
        cv_str = f"{cv_mean:.4f}" if isinstance(cv_mean, float) else "N/A"
        biz_val = info.get("business_value")
        biz_str = f"${biz_val:,.0f}" if isinstance(biz_val, float) else "N/A"

        table.add_row(
            model_key,
            str(info.get("best_model", "N/A")),
            primary_str,
            cv_str,
            biz_str,
        )

    console.print(table)

    # Save comparison JSON
    out_path = Path("data/outputs")
    out_path.mkdir(parents=True, exist_ok=True)
    comparison_file = out_path / "model_comparison.json"

    def _safe_json(obj: object) -> object:
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj

    comparison_serialisable = {
        k: {kk: _safe_json(vv) for kk, vv in v.items()}
        for k, v in comparison.items()
    }
    with open(comparison_file, "w") as f:
        json.dump(comparison_serialisable, f, indent=2, default=str)
    console.print(f"\n  [green]✓[/green] Model comparison saved → {comparison_file}")
    console.print(
        f"\n[bold green]Training complete![/bold green] "
        f"MLflow runs logged to '{tracking_uri}/'"
    )


if __name__ == "__main__":
    app()
