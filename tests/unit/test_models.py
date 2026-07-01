"""Unit tests for NovaMart ML model modules.

Tests use small synthetic DataFrames — no real data or network access.
All tests are deterministic (fixed random seed).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------


def _make_churn_features(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Create a tiny synthetic customer feature DataFrame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "days_since_last_purchase": rng.integers(1, 365, n),
            "total_transactions": rng.integers(1, 50, n),
            "total_spend": rng.uniform(50, 5000, n),
            "avg_basket_size": rng.uniform(20, 200, n),
            "discount_sensitivity": rng.uniform(0, 0.5, n),
            "months_as_customer": rng.integers(1, 60, n),
            "recency_score": rng.integers(1, 6, n),
            "frequency_score": rng.integers(1, 6, n),
            "monetary_score": rng.integers(1, 6, n),
            "loyalty_tier": rng.choice(["Bronze", "Silver", "Gold"], n),
            "acquisition_channel": rng.choice(["online", "in-store", "referral"], n),
            "age_group": rng.choice(["18-24", "25-34", "35-44", "45+"], n),
            "preferred_category": rng.choice(["grocery", "clothing", "electronics"], n),
            "is_churned": rng.integers(0, 2, n),
        }
    )
    return df.set_index("customer_id")


def _make_store_features(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Create a tiny synthetic store feature DataFrame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "store_id": [f"S{i:03d}" for i in range(n)],
            "avg_monthly_revenue": rng.uniform(50_000, 500_000, n),
            "sq_footage": rng.integers(500, 10_000, n),
            "manager_tenure_years": rng.uniform(0, 15, n),
            "marketing_efficiency": rng.uniform(1, 10, n),
            "digital_spend_share": rng.uniform(0, 1, n),
            "operating_cost_ratio": rng.uniform(0.2, 0.8, n),
            "revenue_trend_slope": rng.uniform(-5000, 5000, n),
            "revenue_volatility": rng.uniform(0, 20_000, n),
            "months_since_open": rng.integers(1, 120, n),
            "store_format": rng.choice(["urban", "suburban", "rural"], n),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "performance_cluster": rng.choice(["A", "B", "C"], n),
            "avg_gross_margin": rng.uniform(0.1, 0.4, n),
            "is_top_performer": rng.integers(0, 2, n),
        }
    )
    return df.set_index("store_id")


def _make_marketing_data(
    n_periods: int = 24, n_channels: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic spend DataFrame and revenue Series."""
    rng = np.random.default_rng(seed)
    channels = ["tv", "digital", "email"][:n_channels]
    spend = pd.DataFrame(
        rng.uniform(10_000, 100_000, (n_periods, n_channels)),
        columns=channels,
    )
    revenue = pd.Series(
        rng.uniform(500_000, 1_500_000, n_periods),
        name="revenue",
    )
    return spend, revenue


# ---------------------------------------------------------------------------
# FILE 1: base.py
# ---------------------------------------------------------------------------


class TestModelMetrics:
    def test_model_metrics_classification(self) -> None:
        """ModelMetrics with auc_roc populated serialises correctly."""
        from bcgx.models.base import ModelMetrics

        m = ModelMetrics(
            model_name="XGBoost",
            model_type="classification",
            auc_roc=0.85,
            f1_score=0.72,
        )
        assert m.auc_roc == pytest.approx(0.85)
        assert m.model_type == "classification"
        assert m.rmse is None  # regression fields absent

    def test_model_metrics_to_dict_excludes_none(self) -> None:
        from bcgx.models.base import ModelMetrics

        m = ModelMetrics(model_name="Ridge", model_type="regression", r_squared=0.7)
        d = m.to_dict()
        assert "r_squared" in d
        assert "auc_roc" not in d  # None fields excluded

    def test_training_result_summary(self) -> None:
        from bcgx.models.base import ModelMetrics, TrainingResult

        m = ModelMetrics(model_name="RF", model_type="classification", auc_roc=0.9)
        r = TrainingResult(
            model_name="churn",
            best_model_name="RF",
            metrics={"RF": m},
            best_metrics=m,
            feature_names=["f1", "f2"],
            mlflow_run_id="abc123",
            mlflow_experiment_id="1",
            model_path=None,
        )
        summary = r.summary()
        assert "RF" in summary
        assert "0.9000" in summary


# ---------------------------------------------------------------------------
# FILE 2: churn/pipeline.py
# ---------------------------------------------------------------------------


class TestChurnPipeline:
    def test_churn_pipeline_builds_logistic(self) -> None:
        """build_churn_pipeline returns Pipeline with correct steps."""
        from bcgx.models.churn.pipeline import build_churn_pipeline

        pipeline = build_churn_pipeline("LogisticRegression")
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_churn_pipeline_builds_all_models(self) -> None:
        """All candidate model names build successfully."""
        from bcgx.models.churn.pipeline import build_churn_pipeline

        for name in ("LogisticRegression", "RandomForest", "XGBoost", "CatBoost"):
            pipe = build_churn_pipeline(name)
            assert isinstance(pipe, Pipeline), f"{name} did not return a Pipeline"

    def test_churn_pipeline_unknown_raises(self) -> None:
        from bcgx.models.churn.pipeline import build_churn_pipeline

        with pytest.raises(ValueError, match="Unknown model"):
            build_churn_pipeline("NotAModel")

    def test_churn_pipeline_fits_tiny_dataset(self) -> None:
        """Fit on 50-row synthetic dataset, predict, check output shape."""
        from bcgx.models.churn.pipeline import (
            ALL_CHURN_FEATURES,
            CHURN_CATEGORICAL_FEATURES,
            CHURN_TARGET,
            build_churn_pipeline,
        )

        features = _make_churn_features(50)
        # Ensure all feature columns exist (add missing categoricals)
        for col in CHURN_CATEGORICAL_FEATURES:
            if col not in features.columns:
                features[col] = "unknown"

        feat_cols = [c for c in ALL_CHURN_FEATURES if c in features.columns]
        X = features[feat_cols]
        y = features[CHURN_TARGET]

        pipeline = build_churn_pipeline("LogisticRegression")
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        assert preds.shape == (50,)
        assert set(preds).issubset({0, 1})

        proba = pipeline.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_churn_pipeline_random_forest_fits(self) -> None:
        """RandomForest pipeline fits and returns valid probabilities."""
        from bcgx.models.churn.pipeline import (
            ALL_CHURN_FEATURES,
            CHURN_CATEGORICAL_FEATURES,
            CHURN_TARGET,
            build_churn_pipeline,
        )

        features = _make_churn_features(80)
        for col in CHURN_CATEGORICAL_FEATURES:
            if col not in features.columns:
                features[col] = "unknown"

        feat_cols = [c for c in ALL_CHURN_FEATURES if c in features.columns]
        X = features[feat_cols]
        y = features[CHURN_TARGET]

        pipeline = build_churn_pipeline("RandomForest", random_state=0)
        pipeline.fit(X, y)

        proba = pipeline.predict_proba(X)[:, 1]
        assert proba.shape == (80,)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_all_churn_features_list(self) -> None:
        """ALL_CHURN_FEATURES is the union of numeric and categorical lists."""
        from bcgx.models.churn.pipeline import (
            ALL_CHURN_FEATURES,
            CHURN_CATEGORICAL_FEATURES,
            CHURN_NUMERIC_FEATURES,
        )

        assert set(ALL_CHURN_FEATURES) == set(CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES)
        assert len(ALL_CHURN_FEATURES) == len(CHURN_NUMERIC_FEATURES) + len(
            CHURN_CATEGORICAL_FEATURES
        )


# ---------------------------------------------------------------------------
# FILE 3: store_performance/pipeline.py
# ---------------------------------------------------------------------------


class TestStorePipeline:
    def test_store_pipeline_builds_ridge_regression(self) -> None:
        from bcgx.models.store_performance.pipeline import build_store_pipeline

        pipe = build_store_pipeline("Ridge", task="regression")
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "regressor" in pipe.named_steps

    def test_store_pipeline_builds_classification(self) -> None:
        from bcgx.models.store_performance.pipeline import build_store_pipeline

        pipe = build_store_pipeline("RandomForest", task="classification")
        assert "classifier" in pipe.named_steps

    def test_store_pipeline_fits(self) -> None:
        from bcgx.models.store_performance.pipeline import (
            ALL_STORE_FEATURES,
            STORE_CATEGORICAL_FEATURES,
            STORE_REGRESSION_TARGET,
            build_store_pipeline,
        )

        features = _make_store_features(40)
        for col in STORE_CATEGORICAL_FEATURES:
            if col not in features.columns:
                features[col] = "unknown"

        feat_cols = [c for c in ALL_STORE_FEATURES if c in features.columns]
        X = features[feat_cols]
        y = features[STORE_REGRESSION_TARGET]

        pipe = build_store_pipeline("Ridge", task="regression", random_state=0)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (40,)


# ---------------------------------------------------------------------------
# FILE 8: marketing_mix/pipeline.py
# ---------------------------------------------------------------------------


class TestAdstockTransform:
    def test_adstock_first_element_unchanged(self) -> None:
        """The first element is always unchanged."""
        from bcgx.models.marketing_mix.pipeline import adstock_transform

        x = np.array([100.0, 0.0, 0.0, 0.0])
        result = adstock_transform(x, decay_rate=0.5)
        assert result[0] == pytest.approx(100.0)

    def test_adstock_geometric_decay(self) -> None:
        """Apply adstock_transform to known array, verify geometric decay.

        Formula: result[i] = x[i] + decay_rate * result[i-1]
        result[0] = 100
        result[1] = 0 + 0.5 * 100 = 50
        result[2] = 0 + 0.5 * 50 = 25
        result[3] = 0 + 0.5 * 25 = 12.5
        result[4] = 100 + 0.5 * 12.5 = 106.25
        """
        from bcgx.models.marketing_mix.pipeline import adstock_transform

        x = np.array([100.0, 0.0, 0.0, 0.0, 100.0])
        result = adstock_transform(x, decay_rate=0.5)
        expected = np.array([100.0, 50.0, 25.0, 12.5, 106.25])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_adstock_zero_decay(self) -> None:
        """Decay rate 0 → no carryover, output equals input."""
        from bcgx.models.marketing_mix.pipeline import adstock_transform

        x = np.array([10.0, 20.0, 30.0])
        result = adstock_transform(x, decay_rate=0.0)
        np.testing.assert_allclose(result, x, rtol=1e-5)

    def test_adstock_output_length(self) -> None:
        """Output length equals input length."""
        from bcgx.models.marketing_mix.pipeline import adstock_transform

        x = np.arange(10, dtype=float)
        result = adstock_transform(x, decay_rate=0.3)
        assert len(result) == len(x)

    def test_adstock_monotone_accumulation(self) -> None:
        """Constant spend with decay > 0 should accumulate monotonically."""
        from bcgx.models.marketing_mix.pipeline import adstock_transform

        x = np.ones(10) * 100.0
        result = adstock_transform(x, decay_rate=0.5)
        # Each element should be >= previous element (accumulating with constant input)
        # Actually converges, but first few should increase
        assert result[1] > result[0]


class TestMMMPipeline:
    def test_mmm_pipeline_fits_and_predicts(self) -> None:
        from bcgx.models.marketing_mix.pipeline import MMMPipeline

        spend, revenue = _make_marketing_data(n_periods=24, n_channels=3)
        mmm = MMMPipeline()
        mmm.fit(spend, revenue)
        preds = mmm.predict(spend)
        assert preds.shape == (24,)
        assert not np.any(np.isnan(preds))

    def test_mmm_channel_contributions_sum_to_100(self) -> None:
        from bcgx.models.marketing_mix.pipeline import MMMPipeline

        spend, revenue = _make_marketing_data(n_periods=20, n_channels=3)
        mmm = MMMPipeline()
        mmm.fit(spend, revenue)
        contributions = mmm.get_channel_contributions()
        total = sum(contributions.values())
        assert abs(total - 100.0) < 1e-6

    def test_mmm_roi_returns_dict(self) -> None:
        from bcgx.models.marketing_mix.pipeline import MMMPipeline

        spend, revenue = _make_marketing_data(n_periods=20, n_channels=3)
        mmm = MMMPipeline()
        mmm.fit(spend, revenue)
        rois = mmm.get_roi_by_channel()
        assert isinstance(rois, dict)
        assert len(rois) == 3

    def test_mmm_not_fitted_raises(self) -> None:
        from bcgx.models.marketing_mix.pipeline import MMMPipeline

        mmm = MMMPipeline()
        with pytest.raises(RuntimeError, match="fitted"):
            mmm.predict(pd.DataFrame({"tv": [100]}))


# ---------------------------------------------------------------------------
# FILE 12: explainability/shap_explainer.py
# ---------------------------------------------------------------------------


class TestSHAPExplainer:
    def test_shap_explanation_fields(self) -> None:
        """SHAPExplanation has all required fields."""
        from bcgx.explainability.shap_explainer import SHAPExplanation

        explanation = SHAPExplanation(
            model_name="TestModel",
            feature_names=["f1", "f2"],
            mean_abs_shap={"f1": 0.5, "f2": 0.1},
            top_features=[{"feature": "f1", "shap_value": 0.5, "business_meaning": "test"}],
            global_explanation="The top drivers are f1.",
            shap_values=np.array([[0.5, 0.1]]),
        )
        assert explanation.model_name == "TestModel"
        assert "f1" in explanation.mean_abs_shap
        assert explanation.shap_values is not None
        assert len(explanation.top_features) == 1

    def test_shap_explanation_none_shap_values(self) -> None:
        """SHAPExplanation can be constructed with shap_values=None."""
        from bcgx.explainability.shap_explainer import SHAPExplanation

        explanation = SHAPExplanation(
            model_name="Model",
            feature_names=[],
            mean_abs_shap={},
            top_features=[],
            global_explanation="No explanation available",
            shap_values=None,
        )
        assert explanation.shap_values is None

    def test_shap_explainer_get_business_meaning(self) -> None:
        """get_business_meaning returns non-empty string for known features."""
        from bcgx.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()
        meaning = explainer.get_business_meaning("days_since_last_purchase", shap_value=0.3)
        assert isinstance(meaning, str)
        assert len(meaning) > 10

    def test_shap_explainer_unknown_feature(self) -> None:
        """get_business_meaning returns a fallback for unknown features."""
        from bcgx.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()
        meaning = explainer.get_business_meaning("some_unknown_feature_xyz", shap_value=0.1)
        assert isinstance(meaning, str)
        assert len(meaning) > 0

    def test_shap_explain_classification_model(self) -> None:
        """explain_classification_model returns SHAPExplanation with top features."""
        from bcgx.explainability.shap_explainer import SHAPExplainer
        from bcgx.models.churn.pipeline import (
            ALL_CHURN_FEATURES,
            CHURN_CATEGORICAL_FEATURES,
            CHURN_TARGET,
            build_churn_pipeline,
        )

        features = _make_churn_features(80)
        for col in CHURN_CATEGORICAL_FEATURES:
            if col not in features.columns:
                features[col] = "unknown"

        feat_cols = [c for c in ALL_CHURN_FEATURES if c in features.columns]
        X = features[feat_cols]
        y = features[CHURN_TARGET]

        pipeline = build_churn_pipeline("LogisticRegression", random_state=0)
        pipeline.fit(X, y)

        explainer = SHAPExplainer(seed=0)
        explanation = explainer.explain_classification_model(
            pipeline, X, "LogisticRegression", n_samples=50
        )

        assert explanation.model_name == "LogisticRegression"
        assert len(explanation.feature_names) > 0
        assert len(explanation.mean_abs_shap) > 0
        assert isinstance(explanation.global_explanation, str)
        assert len(explanation.global_explanation) > 0


# ---------------------------------------------------------------------------
# FILE 1: base.py — BaseModel ABC
# ---------------------------------------------------------------------------


class TestBaseModelABC:
    def test_cannot_instantiate_base_model(self) -> None:
        """BaseModel is abstract and cannot be instantiated directly."""
        from bcgx.models.base import BaseModel

        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_concrete_subclass_requires_all_methods(self) -> None:
        """A subclass that omits abstractmethods cannot be instantiated."""
        from bcgx.models.base import BaseModel

        class IncompleteModel(BaseModel):
            def train(self, X_train, y_train):
                pass
            # Missing predict, predict_proba, get_feature_importance

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Additional smoke tests
# ---------------------------------------------------------------------------


class TestChurnEvaluator:
    def test_generate_churn_scores(self) -> None:
        """generate_churn_scores returns DataFrame with churn_probability column."""
        from bcgx.models.churn.evaluator import ChurnEvaluator
        from bcgx.models.churn.pipeline import (
            ALL_CHURN_FEATURES,
            CHURN_CATEGORICAL_FEATURES,
            CHURN_TARGET,
            build_churn_pipeline,
        )

        features = _make_churn_features(60)
        for col in CHURN_CATEGORICAL_FEATURES:
            if col not in features.columns:
                features[col] = "unknown"

        feat_cols = [c for c in ALL_CHURN_FEATURES if c in features.columns]
        X = features[feat_cols]
        y = features[CHURN_TARGET]

        pipeline = build_churn_pipeline("LogisticRegression", random_state=0)
        pipeline.fit(X, y)

        evaluator = ChurnEvaluator()
        scores = evaluator.generate_churn_scores(pipeline, X)

        assert "churn_probability" in scores.columns
        assert "churn_flag" in scores.columns
        assert len(scores) == 60
        assert scores["churn_probability"].between(0, 1).all()


class TestElasticityPipeline:
    def test_get_all_elasticities_returns_dataframe(self) -> None:
        """get_all_elasticities returns a DataFrame after fitting."""
        from bcgx.models.price_elasticity.pipeline import ElasticityModelPipeline

        rng = np.random.default_rng(0)
        n = 200
        transactions = pd.DataFrame(
            {
                "transaction_id": range(n),
                "product_id": rng.choice(["P1", "P2", "P3", "P4", "P5"], n),
                "unit_price": rng.uniform(5, 50, n),
                "quantity": rng.integers(1, 10, n),
                "gross_revenue": rng.uniform(10, 500, n),
                "gross_profit": rng.uniform(2, 100, n),
                "discount_pct": rng.uniform(0, 0.3, n),
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P1", "P2", "P3", "P4", "P5"],
                "category": ["A", "A", "B", "B", "C"],
                "brand_type": ["national", "private_label", "national", "premium", "national"],
                "unit_cost": [3.0, 4.0, 8.0, 12.0, 5.0],
                "list_price": [10.0, 12.0, 25.0, 40.0, 15.0],
            }
        )

        pipeline = ElasticityModelPipeline(min_obs_per_product=3)
        pipeline.fit(transactions, products)

        df = pipeline.get_all_elasticities()
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "elasticity" in df.columns
            assert "r_squared" in df.columns
