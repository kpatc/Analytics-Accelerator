"""Unit tests for the NovaMart synthetic data generator.

Tests use small n values (10-100) to run fast without generating full datasets.
Signal correctness (margins, tier distributions, cluster distributions) is tested
at the statistical level.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from bcgx.data.generator import (
    generate_stores,
    generate_customers,
    generate_products,
)


# ── generate_stores ───────────────────────────────────────────────────────────


class TestGenerateStoresSchema:
    """Schema and column correctness tests for generate_stores."""

    def test_generate_stores_schema(self) -> None:
        """generate_stores with n_stores=10 must have the correct columns."""
        df = generate_stores(seed=42, n_stores=10)
        required_columns = [
            "store_id",
            "region",
            "store_format",
            "sq_footage",
            "open_date",
            "manager_tenure_years",
            "performance_cluster",
            "cluster_profit_multiplier",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_generate_stores_row_count(self) -> None:
        """generate_stores must return exactly n_stores rows."""
        for n in [10, 50, 100]:
            df = generate_stores(seed=42, n_stores=n)
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_generate_stores_store_id_format(self) -> None:
        """store_id must follow 'Sxxx' format and be unique."""
        df = generate_stores(seed=42, n_stores=20)
        assert df["store_id"].is_unique, "store_id values are not unique"
        assert all(df["store_id"].str.startswith("S")), "store_ids must start with 'S'"

    def test_generate_stores_dtypes(self) -> None:
        """Key columns must have appropriate dtypes."""
        df = generate_stores(seed=42, n_stores=20)
        assert pd.api.types.is_integer_dtype(df["sq_footage"]), "sq_footage must be integer dtype"
        assert pd.api.types.is_float_dtype(df["manager_tenure_years"]), "manager_tenure_years must be float"
        assert pd.api.types.is_float_dtype(df["cluster_profit_multiplier"]), "cluster_profit_multiplier must be float"


class TestGenerateStoresNoNulls:
    """Null-value tests for generate_stores."""

    def test_generate_stores_no_nulls(self) -> None:
        """No key columns should have null values."""
        df = generate_stores(seed=42, n_stores=50)
        key_cols = ["store_id", "region", "store_format", "sq_footage",
                    "open_date", "manager_tenure_years", "performance_cluster",
                    "cluster_profit_multiplier"]
        for col in key_cols:
            null_count = df[col].isna().sum()
            assert null_count == 0, f"Column '{col}' has {null_count} null values"

    def test_generate_stores_valid_store_formats(self) -> None:
        """store_format must only contain valid values."""
        df = generate_stores(seed=42, n_stores=100)
        valid_formats = {"urban", "suburban", "rural"}
        assert set(df["store_format"].unique()).issubset(valid_formats)

    def test_generate_stores_valid_regions(self) -> None:
        """region must only contain valid US region names."""
        df = generate_stores(seed=42, n_stores=100)
        valid_regions = {"Northeast", "Southeast", "Midwest", "Southwest", "West"}
        assert set(df["region"].unique()).issubset(valid_regions)

    def test_generate_stores_valid_clusters(self) -> None:
        """performance_cluster must only be A, B, or C."""
        df = generate_stores(seed=42, n_stores=100)
        valid_clusters = {"A", "B", "C"}
        assert set(df["performance_cluster"].unique()).issubset(valid_clusters)

    def test_generate_stores_sq_footage_positive(self) -> None:
        """sq_footage must be positive for all stores."""
        df = generate_stores(seed=42, n_stores=100)
        assert (df["sq_footage"] > 0).all(), "All sq_footage values must be positive"

    def test_generate_stores_cluster_multiplier_matches_cluster(self) -> None:
        """cluster_profit_multiplier must match the expected value for each cluster."""
        df = generate_stores(seed=42, n_stores=100)
        expected = {"A": 2.8, "B": 1.0, "C": 0.4}
        for cluster, expected_mult in expected.items():
            cluster_df = df[df["performance_cluster"] == cluster]
            if len(cluster_df) > 0:
                assert (cluster_df["cluster_profit_multiplier"] == expected_mult).all(), (
                    f"Cluster {cluster} should have multiplier {expected_mult}"
                )


class TestGenerateStoreClusters:
    """Cluster distribution tests for generate_stores."""

    def test_generate_store_clusters(self) -> None:
        """Cluster distribution: A~15%, B~60%, C~25% within 10% tolerance."""
        df = generate_stores(seed=42, n_stores=800)
        counts = df["performance_cluster"].value_counts(normalize=True)
        tolerance = 0.10

        a_pct = counts.get("A", 0.0)
        b_pct = counts.get("B", 0.0)
        c_pct = counts.get("C", 0.0)

        assert abs(a_pct - 0.15) <= tolerance, f"Cluster A: expected ~15%, got {a_pct:.1%}"
        assert abs(b_pct - 0.60) <= tolerance, f"Cluster B: expected ~60%, got {b_pct:.1%}"
        assert abs(c_pct - 0.25) <= tolerance, f"Cluster C: expected ~25%, got {c_pct:.1%}"

    def test_generate_store_clusters_sums_to_one(self) -> None:
        """All cluster percentages must sum to 1.0."""
        df = generate_stores(seed=42, n_stores=200)
        counts = df["performance_cluster"].value_counts(normalize=True)
        assert abs(counts.sum() - 1.0) < 1e-6


# ── generate_customers ────────────────────────────────────────────────────────


class TestGenerateCustomersTierDistribution:
    """Loyalty tier distribution tests for generate_customers."""

    def test_generate_customers_tier_distribution(self) -> None:
        """Gold ~20%, Silver ~35%, Bronze ~45% within 5% absolute tolerance."""
        df = generate_customers(seed=42, n_customers=10_000)
        counts = df["loyalty_tier"].value_counts(normalize=True)
        tolerance = 0.05

        gold_pct = counts.get("Gold", 0.0)
        silver_pct = counts.get("Silver", 0.0)
        bronze_pct = counts.get("Bronze", 0.0)

        assert abs(gold_pct - 0.20) <= tolerance, f"Gold: expected ~20%, got {gold_pct:.1%}"
        assert abs(silver_pct - 0.35) <= tolerance, f"Silver: expected ~35%, got {silver_pct:.1%}"
        assert abs(bronze_pct - 0.45) <= tolerance, f"Bronze: expected ~45%, got {bronze_pct:.1%}"

    def test_generate_customers_valid_tiers(self) -> None:
        """loyalty_tier must only contain Gold, Silver, or Bronze."""
        df = generate_customers(seed=42, n_customers=1_000)
        valid_tiers = {"Gold", "Silver", "Bronze"}
        assert set(df["loyalty_tier"].unique()).issubset(valid_tiers)

    def test_generate_customers_segment_distribution(self) -> None:
        """Segment: Premium ~15%, Regular ~55%, Occasional ~30% within 5%."""
        df = generate_customers(seed=42, n_customers=10_000)
        counts = df["segment"].value_counts(normalize=True)
        tolerance = 0.05

        assert abs(counts.get("Premium", 0.0) - 0.15) <= tolerance
        assert abs(counts.get("Regular", 0.0) - 0.55) <= tolerance
        assert abs(counts.get("Occasional", 0.0) - 0.30) <= tolerance

    def test_generate_customers_no_null_customer_id(self) -> None:
        """customer_id must not have any null values."""
        df = generate_customers(seed=42, n_customers=500)
        assert df["customer_id"].isna().sum() == 0

    def test_generate_customers_unique_ids(self) -> None:
        """customer_id must be unique across all rows."""
        df = generate_customers(seed=42, n_customers=1_000)
        assert df["customer_id"].is_unique

    def test_generate_customers_email_opt_in_is_bool(self) -> None:
        """email_opt_in must be a boolean column."""
        df = generate_customers(seed=42, n_customers=500)
        assert df["email_opt_in"].dtype == bool or set(df["email_opt_in"].unique()).issubset({True, False})


# ── generate_products ─────────────────────────────────────────────────────────


class TestGenerateProductsMargin:
    """Margin and pricing tests for generate_products."""

    def test_generate_products_margin(self) -> None:
        """Private label must have higher list_price/unit_cost ratio than national brand."""
        df = generate_products(seed=42, n_products=500)

        pl = df[df["brand_type"] == "private_label"]
        nb = df[df["brand_type"] == "national_brand"]

        assert len(pl) > 0, "No private_label products found"
        assert len(nb) > 0, "No national_brand products found"

        pl_ratio = (pl["list_price"] / pl["unit_cost"]).mean()
        nb_ratio = (nb["list_price"] / nb["unit_cost"]).mean()

        assert pl_ratio > nb_ratio, (
            f"Private label price/cost ratio ({pl_ratio:.3f}) must exceed "
            f"national brand ({nb_ratio:.3f})"
        )

    def test_generate_products_private_label_markup(self) -> None:
        """Private label markup should be approximately 2.3x cost."""
        df = generate_products(seed=42, n_products=500)
        pl = df[df["brand_type"] == "private_label"]
        ratios = pl["list_price"] / pl["unit_cost"]
        # Allow 1% tolerance around 2.3 (it's set exactly in generator)
        assert abs(ratios.mean() - 2.3) < 0.05, f"PL markup avg: {ratios.mean():.3f} (expected ~2.3)"

    def test_generate_products_national_brand_markup(self) -> None:
        """National brand markup should be approximately 1.6x cost."""
        df = generate_products(seed=42, n_products=500)
        nb = df[df["brand_type"] == "national_brand"]
        ratios = nb["list_price"] / nb["unit_cost"]
        assert abs(ratios.mean() - 1.6) < 0.05, f"NB markup avg: {ratios.mean():.3f} (expected ~1.6)"

    def test_generate_products_all_prices_positive(self) -> None:
        """All unit_cost and list_price values must be positive."""
        df = generate_products(seed=42, n_products=200)
        assert (df["unit_cost"] > 0).all(), "Some unit_cost values are not positive"
        assert (df["list_price"] > 0).all(), "Some list_price values are not positive"

    def test_generate_products_list_price_exceeds_cost(self) -> None:
        """list_price must always exceed unit_cost (positive margin)."""
        df = generate_products(seed=42, n_products=200)
        assert (df["list_price"] >= df["unit_cost"]).all(), (
            "Some products have list_price < unit_cost"
        )

    def test_generate_products_no_null_ids(self) -> None:
        """product_id must have no nulls and be unique."""
        df = generate_products(seed=42, n_products=100)
        assert df["product_id"].isna().sum() == 0
        assert df["product_id"].is_unique

    def test_generate_products_valid_categories(self) -> None:
        """category must only contain expected NovaMart categories."""
        df = generate_products(seed=42, n_products=200)
        valid_categories = {
            "Electronics", "Apparel", "Home & Garden", "Food & Beverage",
            "Health & Beauty", "Sports", "Toys",
        }
        assert set(df["category"].unique()).issubset(valid_categories)

    def test_generate_products_valid_brand_types(self) -> None:
        """brand_type must only be 'private_label' or 'national_brand'."""
        df = generate_products(seed=42, n_products=100)
        valid_brand_types = {"private_label", "national_brand"}
        assert set(df["brand_type"].unique()).issubset(valid_brand_types)

    def test_generate_products_elasticity_ranges(self) -> None:
        """price_elasticity_coefficient must be negative for all categories."""
        df = generate_products(seed=42, n_products=200)
        assert (df["price_elasticity_coefficient"] < 0).all(), (
            "All price elasticity coefficients must be negative"
        )

    def test_generate_products_electronics_more_elastic(self) -> None:
        """Electronics should be more price-elastic (more negative) than food."""
        df = generate_products(seed=42, n_products=1_000)
        electronics = df[df["category"] == "Electronics"]["price_elasticity_coefficient"]
        food = df[df["category"] == "Food & Beverage"]["price_elasticity_coefficient"]
        if len(electronics) > 0 and len(food) > 0:
            assert electronics.mean() < food.mean(), (
                "Electronics should be more elastic (more negative coefficient) than Food"
            )


# ── Reproducibility ───────────────────────────────────────────────────────────


class TestReproducibility:
    """Verify that the same seed always produces identical output."""

    def test_generate_stores_reproducible(self) -> None:
        """Same seed must produce identical stores DataFrames."""
        df1 = generate_stores(seed=99, n_stores=20)
        df2 = generate_stores(seed=99, n_stores=20)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_customers_reproducible(self) -> None:
        """Same seed must produce identical customers DataFrames."""
        df1 = generate_customers(seed=99, n_customers=500)
        df2 = generate_customers(seed=99, n_customers=500)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_products_reproducible(self) -> None:
        """Same seed must produce identical products DataFrames."""
        df1 = generate_products(seed=99, n_products=100)
        df2 = generate_products(seed=99, n_products=100)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_output(self) -> None:
        """Different seeds must produce different store_id sequences."""
        df1 = generate_stores(seed=1, n_stores=50)
        df2 = generate_stores(seed=2, n_stores=50)
        # store_ids are deterministic (just range), so check a different column
        assert not (df1["region"] == df2["region"]).all(), (
            "Different seeds should produce different regional distributions"
        )
