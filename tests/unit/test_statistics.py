"""Unit tests for the NovaMart statistics and EDA modules.

Tests use small synthetic DataFrames to validate:
1. HypothesisTester returns properly structured HypothesisTestResult
2. RFM scoring returns scores in the valid range [1, 5]
3. ElasticityAnalyzer returns negative elasticity for elastic categories
4. ABTestAnalyzer returns ABTestResult with all required fields

All tests are self-contained and do not require the real parquet data files.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bcgx.statistics.hypothesis import HypothesisTester, HypothesisTestResult
from bcgx.statistics.segmentation import SegmentationAnalyzer
from bcgx.statistics.elasticity import ElasticityAnalyzer, ElasticityEstimate
from bcgx.statistics.ab_testing import ABTestAnalyzer, ABTestResult


# ------------------------------------------------------------------ #
# Synthetic data factories                                            #
# ------------------------------------------------------------------ #

def _make_transactions(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a minimal synthetic transaction DataFrame."""
    rng = np.random.default_rng(seed)
    store_ids = [f"S{i:03d}" for i in range(1, 11)]  # 10 stores
    customer_ids = [f"C{i:06d}" for i in range(1, 51)]  # 50 customers
    product_ids = [f"P{i:05d}" for i in range(1, 21)]  # 20 products
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")

    rows = []
    for i in range(n):
        date = dates[rng.integers(0, len(dates))]
        unit_price = float(rng.uniform(5.0, 200.0))
        discount_pct = float(rng.uniform(0.0, 0.40))
        quantity = int(rng.integers(1, 10))
        gross_revenue = unit_price * (1 - discount_pct) * quantity
        cogs = gross_revenue * rng.uniform(0.40, 0.75)
        gross_profit = gross_revenue - cogs
        rows.append(
            {
                "transaction_id": f"T{i:09d}",
                "date": date,
                "year_month": date.strftime("%Y-%m"),
                "store_id": store_ids[rng.integers(0, len(store_ids))],
                "store_format": rng.choice(["urban", "suburban", "rural"]),
                "customer_id": customer_ids[rng.integers(0, len(customer_ids))],
                "product_id": product_ids[rng.integers(0, len(product_ids))],
                "quantity": quantity,
                "discount_pct": discount_pct,
                "unit_price": unit_price,
                "gross_revenue": gross_revenue,
                "cogs": cogs,
                "gross_profit": gross_profit,
            }
        )
    return pd.DataFrame(rows)


def _make_stores(n: int = 10) -> pd.DataFrame:
    """Generate a minimal synthetic stores DataFrame."""
    return pd.DataFrame(
        {
            "store_id": [f"S{i:03d}" for i in range(1, n + 1)],
            "city": [f"City{i}" for i in range(1, n + 1)],
            "state": ["CA"] * n,
            "region": (["North"] * (n // 2) + ["South"] * (n - n // 2)),
            "store_format": (["urban"] * (n // 3) + ["suburban"] * (n // 3) + ["rural"] * (n - 2 * (n // 3))),
            "sq_footage": np.random.default_rng(0).integers(2000, 20000, size=n).tolist(),
            "open_date": ["2020-01-01"] * n,
            "manager_tenure_years": np.random.default_rng(1).uniform(0.5, 10.0, size=n).tolist(),
            "performance_cluster": (["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n - 2 * (n // 3))),
            "cluster_profit_multiplier": [1.2] * (n // 3) + [1.0] * (n // 3) + [0.8] * (n - 2 * (n // 3)),
        }
    )


def _make_customers(n: int = 50) -> pd.DataFrame:
    """Generate a minimal synthetic customers DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
            "segment": rng.choice(["Regular", "Premium"], size=n).tolist(),
            "loyalty_tier": rng.choice(["Bronze", "Silver", "Gold"], size=n).tolist(),
            "acquisition_channel": rng.choice(["online", "in-store", "referral"], size=n).tolist(),
            "region": rng.choice(["North", "South"], size=n).tolist(),
            "acquisition_date": ["2021-01-01"] * n,
            "email_opt_in": rng.choice([True, False], size=n).tolist(),
            "age_group": rng.choice(["18-24", "25-34", "35-49", "50-64"], size=n).tolist(),
        }
    )


def _make_products(n: int = 20) -> pd.DataFrame:
    """Generate a minimal synthetic products DataFrame."""
    rng = np.random.default_rng(7)
    categories = ["Electronics", "Food", "Apparel", "Home"]
    brand_types = ["private_label", "national_brand"]
    return pd.DataFrame(
        {
            "product_id": [f"P{i:05d}" for i in range(1, n + 1)],
            "product_name": [f"Product {i}" for i in range(1, n + 1)],
            "category": [categories[i % len(categories)] for i in range(n)],
            "subcategory": ["sub"] * n,
            "brand_type": [brand_types[i % 2] for i in range(n)],
            "unit_cost": rng.uniform(5.0, 80.0, size=n).tolist(),
            "list_price": rng.uniform(10.0, 150.0, size=n).tolist(),
            "gross_margin_pct": rng.uniform(20.0, 60.0, size=n).tolist(),
            "price_elasticity_coefficient": rng.uniform(-2.5, -0.5, size=n).tolist(),
        }
    )


def _make_marketing(stores: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Generate a minimal synthetic marketing spend DataFrame."""
    rng = np.random.default_rng(42)
    rows = []
    for _, row in stores.iterrows():
        for m in range(months):
            date = pd.Timestamp("2022-01-01") + pd.DateOffset(months=m)
            for channel in ["digital", "tv", "print"]:
                rows.append(
                    {
                        "store_id": row["store_id"],
                        "year_month": date.strftime("%Y-%m"),
                        "store_format": row["store_format"],
                        "channel": channel,
                        "spend_usd": float(rng.uniform(1000.0, 50000.0)),
                        "digital_roi_multiplier": float(rng.uniform(1.2, 3.5)) if channel == "digital" else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Test 1: HypothesisTester returns well-formed result                 #
# ------------------------------------------------------------------ #

class TestHypothesisTester:
    """Validate that HypothesisTester returns properly structured results."""

    @pytest.fixture()
    def synthetic_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tx = _make_transactions(n=600)
        stores = _make_stores(n=10)
        customers = _make_customers(n=50)
        products = _make_products(n=20)
        marketing = _make_marketing(stores, months=12)
        return tx, stores, customers, products, marketing

    def test_hypothesis_tester_returns_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """HypothesisTester.test_cluster_revenue_difference returns a HypothesisTestResult."""
        tx, stores, customers, products, marketing = synthetic_data
        tester = HypothesisTester()
        result = tester.test_cluster_revenue_difference(tx, stores)

        assert isinstance(result, HypothesisTestResult)
        assert result.hypothesis_id == "H1"
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.rejected, bool)
        assert result.alpha == pytest.approx(0.05)
        assert isinstance(result.effect_size, float)
        assert result.effect_size_interpretation in {"negligible", "small", "medium", "large"}
        assert len(result.business_conclusion) > 50
        assert len(result.statistical_conclusion) > 20

    def test_hypothesis_tester_discount_margin(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """HypothesisTester.test_discount_margin_correlation returns a valid result."""
        tx, *_ = synthetic_data
        tester = HypothesisTester()
        result = tester.test_discount_margin_correlation(tx)

        assert isinstance(result, HypothesisTestResult)
        assert result.hypothesis_id == "H5"
        assert not math.isnan(result.p_value)
        assert 0.0 <= result.p_value <= 1.0
        assert result.effect_size >= 0.0
        assert result.test_name == "Pearson correlation test"

    def test_private_label_margin(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """HypothesisTester.test_private_label_margin returns a valid H3 result."""
        tx, stores, customers, products, marketing = synthetic_data
        tester = HypothesisTester()
        result = tester.test_private_label_margin(tx, products)

        assert isinstance(result, HypothesisTestResult)
        assert result.hypothesis_id == "H3"
        assert 0.0 <= result.p_value <= 1.0 or math.isnan(result.p_value)


# ------------------------------------------------------------------ #
# Test 2: RFM scoring in valid range                                   #
# ------------------------------------------------------------------ #

class TestRFMScoring:
    """Validate RFM score range and segment label assignment."""

    def test_rfm_scoring_range(self) -> None:
        """RFM scores must be integers in the range [1, 5]."""
        tx = _make_transactions(n=400)
        analyzer = SegmentationAnalyzer()
        rfm_df = analyzer.compute_rfm(tx)

        assert "recency_score" in rfm_df.columns
        assert "frequency_score" in rfm_df.columns
        assert "monetary_score" in rfm_df.columns

        assert rfm_df["recency_score"].between(1, 5).all(), "Recency scores must be in [1, 5]"
        assert rfm_df["frequency_score"].between(1, 5).all(), "Frequency scores must be in [1, 5]"
        assert rfm_df["monetary_score"].between(1, 5).all(), "Monetary scores must be in [1, 5]"

    def test_rfm_scoring_has_segment_label(self) -> None:
        """Every RFM row must have a non-null segment_label."""
        tx = _make_transactions(n=300)
        analyzer = SegmentationAnalyzer()
        rfm_df = analyzer.compute_rfm(tx)

        assert "segment_label" in rfm_df.columns
        assert rfm_df["segment_label"].notna().all()
        valid_labels = {
            "Champions",
            "Loyal Customers",
            "Potential Loyalists",
            "Recent Customers",
            "At Risk",
            "Lost",
            "Promising",
            "Needs Attention",
        }
        assert rfm_df["segment_label"].isin(valid_labels).all()

    def test_rfm_with_reference_date(self) -> None:
        """RFM scoring accepts a reference_date string and produces correct recency."""
        tx = _make_transactions(n=200)
        analyzer = SegmentationAnalyzer()
        rfm_df = analyzer.compute_rfm(tx, reference_date="2025-01-01")

        assert len(rfm_df) > 0
        assert rfm_df["recency_days"].min() >= 0


# ------------------------------------------------------------------ #
# Test 3: ElasticityAnalyzer returns negative elasticity              #
# ------------------------------------------------------------------ #

class TestElasticityAnalyzer:
    """Validate elasticity estimation logic."""

    def test_elasticity_negative_for_elastic_category(self) -> None:
        """For a synthetically elastic category, the elasticity coefficient must be negative."""
        # Create data where quantity clearly falls as price rises (elastic)
        rng = np.random.default_rng(123)
        n = 200
        prices = rng.uniform(5.0, 50.0, size=n)
        # quantity decreases sharply with price: elasticity ≈ -2
        quantity = np.clip(
            np.exp(4.0 - 2.0 * np.log(prices) + rng.normal(0, 0.1, size=n)), 1, None
        ).astype(int)
        discount = rng.uniform(0.0, 0.30, size=n)

        tx = pd.DataFrame(
            {
                "transaction_id": [f"T{i}" for i in range(n)],
                "product_id": ["P00001"] * n,
                "quantity": quantity,
                "unit_price": prices,
                "discount_pct": discount,
                "gross_revenue": prices * quantity * (1 - discount),
                "cogs": prices * quantity * (1 - discount) * 0.5,
                "gross_profit": prices * quantity * (1 - discount) * 0.5,
                "date": pd.date_range("2022-01-01", periods=n, freq="D"),
                "year_month": pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y-%m"),
                "store_id": ["S001"] * n,
                "store_format": ["urban"] * n,
                "customer_id": [f"C{i:06d}" for i in range(n)],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P00001"],
                "category": ["Electronics"],
                "subcategory": ["TV"],
                "brand_type": ["national_brand"],
                "unit_cost": [10.0],
                "list_price": [50.0],
                "gross_margin_pct": [40.0],
                "price_elasticity_coefficient": [-2.0],
            }
        )

        analyzer = ElasticityAnalyzer()
        results = analyzer.estimate_category_elasticity(tx, products)

        assert len(results) > 0, "Should produce at least one elasticity estimate"
        estimate = results[0]
        assert isinstance(estimate, ElasticityEstimate)
        assert estimate.elasticity_coefficient < 0, (
            f"Expected negative elasticity, got {estimate.elasticity_coefficient}"
        )
        assert estimate.interpretation in {"elastic", "inelastic", "unit elastic"}
        assert estimate.n_observations > 0
        assert isinstance(estimate.pricing_recommendation, str)
        assert len(estimate.pricing_recommendation) > 20

    def test_elasticity_revenue_impact_elastic(self) -> None:
        """Revenue impact of 10% price increase should be negative for elastic goods."""
        analyzer = ElasticityAnalyzer()
        impact = analyzer._revenue_impact_10pct_increase(-2.0)
        assert impact < 0, f"Expected negative revenue impact for elastic good, got {impact}"

    def test_elasticity_revenue_impact_inelastic(self) -> None:
        """Revenue impact of 10% price increase should be positive for inelastic goods."""
        analyzer = ElasticityAnalyzer()
        impact = analyzer._revenue_impact_10pct_increase(-0.5)
        assert impact > 0, f"Expected positive revenue impact for inelastic good, got {impact}"


# ------------------------------------------------------------------ #
# Test 4: ABTestAnalyzer returns ABTestResult with all required fields #
# ------------------------------------------------------------------ #

class TestABTestAnalyzer:
    """Validate ABTestAnalyzer.analyze_loyalty_fee_impact returns correct structure."""

    @pytest.fixture()
    def ab_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic data with clear Silver/Gold tier split over 36 months."""
        # 100 Silver customers, 100 Gold customers, 36 months
        rng = np.random.default_rng(99)
        rows = []
        dates = pd.date_range("2022-01-01", periods=36, freq="MS")
        for tier, cust_range in [("Silver", range(1, 51)), ("Gold", range(51, 101))]:
            for cust_num in cust_range:
                n_months = rng.integers(5, 37)
                month_indices = rng.choice(len(dates), size=n_months, replace=False)
                for mi in month_indices:
                    date = dates[mi]
                    rows.append(
                        {
                            "transaction_id": f"T{len(rows):09d}",
                            "date": date,
                            "year_month": date.strftime("%Y-%m"),
                            "store_id": "S001",
                            "store_format": "urban",
                            "customer_id": f"C{cust_num:06d}",
                            "product_id": "P00001",
                            "quantity": 1,
                            "discount_pct": 0.10,
                            "unit_price": 50.0,
                            "gross_revenue": 45.0,
                            "cogs": 20.0,
                            "gross_profit": 25.0,
                        }
                    )

        tx = pd.DataFrame(rows)
        customers = pd.DataFrame(
            {
                "customer_id": [f"C{i:06d}" for i in range(1, 101)],
                "loyalty_tier": ["Silver"] * 50 + ["Gold"] * 50,
                "segment": ["Regular"] * 100,
                "acquisition_channel": ["online"] * 100,
                "region": ["North"] * 100,
                "acquisition_date": ["2021-01-01"] * 100,
                "email_opt_in": [True] * 100,
                "age_group": ["25-34"] * 100,
            }
        )
        return tx, customers

    def test_ab_test_result_fields(
        self, ab_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """analyze_loyalty_fee_impact returns ABTestResult with all required fields."""
        tx, customers = ab_data
        analyzer = ABTestAnalyzer()
        result = analyzer.analyze_loyalty_fee_impact(tx, customers)

        assert isinstance(result, ABTestResult)

        # All required fields present and correctly typed
        assert isinstance(result.test_name, str)
        assert isinstance(result.treatment_description, str)
        assert isinstance(result.control_description, str)
        assert isinstance(result.metric, str)
        assert isinstance(result.absolute_lift, float | int) or math.isnan(result.absolute_lift)
        assert isinstance(result.relative_lift_pct, float | int) or math.isnan(result.relative_lift_pct)
        assert isinstance(result.p_value, float) or math.isnan(result.p_value)
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2
        assert isinstance(result.statistical_power, float) or math.isnan(result.statistical_power)
        assert isinstance(result.sample_size_treatment, int)
        assert isinstance(result.sample_size_control, int)
        assert isinstance(result.business_impact, str)
        assert len(result.business_impact) > 30

    def test_ab_test_p_value_in_range(
        self, ab_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """p-value must be in [0, 1] or NaN."""
        tx, customers = ab_data
        result = ABTestAnalyzer().analyze_loyalty_fee_impact(tx, customers)
        if not math.isnan(result.p_value):
            assert 0.0 <= result.p_value <= 1.0

    def test_sample_size_calculation(self) -> None:
        """compute_required_sample_size returns a positive integer."""
        analyzer = ABTestAnalyzer()
        n = analyzer.compute_required_sample_size(
            baseline_rate=3.0,
            minimum_detectable_effect=0.5,
            alpha=0.05,
            power=0.80,
        )
        assert isinstance(n, int)
        assert n > 0
