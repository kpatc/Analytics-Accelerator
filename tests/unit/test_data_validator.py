"""Unit tests for the NovaMart DataValidator.

Tests use small synthetic DataFrames constructed manually — not full dataset generation.
Each test validates a specific check in DataValidator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bcgx.data.validator import DataValidator, ValidationResult


# ── Fixtures: minimal valid DataFrames ───────────────────────────────────────


def _make_valid_stores(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid stores DataFrame."""
    return pd.DataFrame({
        "store_id": [f"S{i:03d}" for i in range(1, n + 1)],
        "city": ["Springfield"] * n,
        "state": ["IL"] * n,
        "region": (["Northeast", "Southeast", "Midwest", "Southwest", "West"] * 10)[:n],
        "store_format": (["urban", "suburban", "rural"] * 10)[:n],
        "sq_footage": [10_000] * n,
        "open_date": ["2015-06-15"] * n,
        "manager_tenure_years": [5.0] * n,
        "performance_cluster": (["A", "B", "C"] * 10)[:n],
        "cluster_profit_multiplier": ([2.8, 1.0, 0.4] * 10)[:n],
    })


def _make_valid_customers(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid customers DataFrame."""
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "segment": (["Premium", "Regular", "Occasional"] * 10)[:n],
        "loyalty_tier": (["Gold", "Silver", "Bronze"] * 10)[:n],
        "acquisition_channel": (["digital", "in-store", "referral", "tv"] * 10)[:n],
        "region": (["Northeast", "Southeast", "Midwest", "Southwest", "West"] * 10)[:n],
        "acquisition_date": ["2022-03-01"] * n,
        "email_opt_in": [True] * n,
        "age_group": (["18-24", "25-34", "35-49", "50-64", "65+"] * 10)[:n],
    })


def _make_valid_products(n: int = 6) -> pd.DataFrame:
    """Return a minimal valid products DataFrame with both brand types."""
    half = n // 2
    return pd.DataFrame({
        "product_id": [f"P{i:05d}" for i in range(1, n + 1)],
        "product_name": [f"Product {i}" for i in range(1, n + 1)],
        "category": (["Electronics", "Food & Beverage"] * 10)[:n],
        "subcategory": (["Smartphones", "Beverages"] * 10)[:n],
        "brand_type": (["private_label"] * half + ["national_brand"] * (n - half)),
        "unit_cost": [10.0] * n,
        # Private label: cost * 2.3 = 23.0; national brand: cost * 1.6 = 16.0
        "list_price": ([23.0] * half + [16.0] * (n - half)),
        "gross_margin_pct": ([56.5] * half + [37.5] * (n - half)),
        "price_elasticity_coefficient": ([-1.9] * half + [-0.5] * (n - half)),
    })


def _make_valid_transactions(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid transactions DataFrame."""
    return pd.DataFrame({
        "transaction_id": [f"T000{i:07d}" for i in range(n)],
        "date": pd.to_datetime(["2022-01-15"] * n),
        "year_month": ["2022-01"] * n,
        "store_id": [f"S{i:03d}" for i in range(1, n + 1)],
        "store_format": ["urban"] * n,
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "product_id": [f"P{i:05d}" for i in range(1, n + 1)],
        "quantity": [2] * n,
        "discount_pct": [0.0] * n,
        "unit_price": [20.0] * n,
        "gross_revenue": [40.0] * n,
        "cogs": [25.0] * n,
        "gross_profit": [15.0] * n,
    })


# ── Validator instance ────────────────────────────────────────────────────────

@pytest.fixture()
def validator() -> DataValidator:
    return DataValidator()


# ── test_valid_stores_passes ──────────────────────────────────────────────────


class TestValidStores:
    def test_valid_stores_passes(self, validator: DataValidator) -> None:
        """Valid store DataFrame should produce ValidationResult.passed = True."""
        df = _make_valid_stores(n=5)
        result = validator.validate_stores(df)
        assert isinstance(result, ValidationResult)
        assert result.table == "stores"
        assert result.passed, (
            f"Validation failed unexpectedly. Failures: {result.failed_checks}"
        )

    def test_valid_stores_all_checks_pass(self, validator: DataValidator) -> None:
        """Every individual check should pass for a valid stores DataFrame."""
        df = _make_valid_stores(n=10)
        result = validator.validate_stores(df)
        for check in result.checks:
            assert check["passed"], f"Check '{check['check']}' failed: {check['detail']}"


# ── test_duplicate_pk_fails ───────────────────────────────────────────────────


class TestDuplicatePKFails:
    def test_duplicate_store_id_fails(self, validator: DataValidator) -> None:
        """Injecting a duplicate store_id must cause validation to fail."""
        df = _make_valid_stores(n=5)
        # Inject duplicate: set row 1 store_id same as row 0
        df.loc[1, "store_id"] = df.loc[0, "store_id"]
        result = validator.validate_stores(df)
        assert not result.passed, "Validation should fail when duplicate store_ids exist"
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_duplicate_store_id" in check_names

    def test_duplicate_customer_id_fails(self, validator: DataValidator) -> None:
        """Injecting a duplicate customer_id must cause customer validation to fail."""
        df = _make_valid_customers(n=5)
        df.loc[2, "customer_id"] = df.loc[0, "customer_id"]
        result = validator.validate_customers(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_duplicate_customer_id" in check_names

    def test_duplicate_product_id_fails(self, validator: DataValidator) -> None:
        """Injecting a duplicate product_id must cause product validation to fail."""
        df = _make_valid_products(n=6)
        df.loc[3, "product_id"] = df.loc[0, "product_id"]
        result = validator.validate_products(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_duplicate_product_id" in check_names

    def test_duplicate_transaction_id_fails(self, validator: DataValidator) -> None:
        """Injecting a duplicate transaction_id must cause transaction validation to fail."""
        df = _make_valid_transactions(n=5)
        df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]
        result = validator.validate_transactions(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_duplicate_transaction_id" in check_names


# ── test_null_pk_fails ────────────────────────────────────────────────────────


class TestNullPKFails:
    def test_null_store_id_fails(self, validator: DataValidator) -> None:
        """Injecting a null store_id must cause validation to fail."""
        df = _make_valid_stores(n=5)
        df.loc[2, "store_id"] = None
        result = validator.validate_stores(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_null_store_id" in check_names

    def test_null_customer_id_fails(self, validator: DataValidator) -> None:
        """Injecting a null customer_id must cause customer validation to fail."""
        df = _make_valid_customers(n=5)
        df.loc[0, "customer_id"] = None
        result = validator.validate_customers(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_null_customer_id" in check_names

    def test_null_product_id_fails(self, validator: DataValidator) -> None:
        """Injecting a null product_id must cause product validation to fail."""
        df = _make_valid_products(n=6)
        df.loc[0, "product_id"] = None
        result = validator.validate_products(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_null_product_id" in check_names

    def test_null_transaction_id_fails(self, validator: DataValidator) -> None:
        """Injecting a null transaction_id must cause transaction validation to fail."""
        df = _make_valid_transactions(n=5)
        df.loc[3, "transaction_id"] = None
        result = validator.validate_transactions(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "no_null_transaction_id" in check_names


# ── test_negative_price_fails ─────────────────────────────────────────────────


class TestNegativePriceFails:
    def test_negative_list_price_fails(self, validator: DataValidator) -> None:
        """Injecting a negative list_price must cause product validation to fail."""
        df = _make_valid_products(n=6)
        df.loc[0, "list_price"] = -5.0
        result = validator.validate_products(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "list_price_positive" in check_names

    def test_zero_list_price_fails(self, validator: DataValidator) -> None:
        """A zero list_price should also fail the positivity check."""
        df = _make_valid_products(n=6)
        df.loc[0, "list_price"] = 0.0
        result = validator.validate_products(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "list_price_positive" in check_names

    def test_negative_unit_cost_fails(self, validator: DataValidator) -> None:
        """Injecting a negative unit_cost must fail the unit_cost_positive check."""
        df = _make_valid_products(n=6)
        df.loc[1, "unit_cost"] = -1.0
        result = validator.validate_products(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "unit_cost_positive" in check_names

    def test_negative_unit_price_in_transactions_fails(self, validator: DataValidator) -> None:
        """Injecting a negative unit_price in transactions must fail."""
        df = _make_valid_transactions(n=5)
        df.loc[0, "unit_price"] = -10.0
        result = validator.validate_transactions(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "unit_price_positive" in check_names


# ── Business rule: gross_profit = gross_revenue - cogs ────────────────────────


class TestGrossProfitBusinessRule:
    def test_gross_profit_mismatch_fails(self, validator: DataValidator) -> None:
        """Injecting a gross_profit inconsistent with gross_revenue - cogs must fail."""
        df = _make_valid_transactions(n=5)
        # Corrupt gross_profit for one row
        df.loc[2, "gross_profit"] = 999.99
        result = validator.validate_transactions(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "gross_profit_equals_revenue_minus_cogs" in check_names

    def test_valid_gross_profit_passes(self, validator: DataValidator) -> None:
        """Transactions with correct gross_profit must pass the business rule check."""
        df = _make_valid_transactions(n=5)
        result = validator.validate_transactions(df)
        assert result.passed, f"Valid transactions failed: {result.failed_checks}"


# ── Categorical validation ────────────────────────────────────────────────────


class TestCategoricalValidation:
    def test_invalid_store_format_fails(self, validator: DataValidator) -> None:
        """An unrecognised store_format value must fail the categorical check."""
        df = _make_valid_stores(n=5)
        df.loc[0, "store_format"] = "megastore"
        result = validator.validate_stores(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "valid_store_format" in check_names

    def test_invalid_loyalty_tier_fails(self, validator: DataValidator) -> None:
        """An unrecognised loyalty_tier must fail the categorical check."""
        df = _make_valid_customers(n=5)
        df.loc[0, "loyalty_tier"] = "Platinum"
        result = validator.validate_customers(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "valid_loyalty_tier" in check_names

    def test_invalid_cluster_fails(self, validator: DataValidator) -> None:
        """An unrecognised performance_cluster value must fail the check."""
        df = _make_valid_stores(n=5)
        df.loc[0, "performance_cluster"] = "D"
        result = validator.validate_stores(df)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "valid_performance_cluster" in check_names


# ── Referential integrity ─────────────────────────────────────────────────────


class TestReferentialIntegrity:
    def test_orphan_store_id_fails(self, validator: DataValidator) -> None:
        """A transaction with a store_id not in the stores table must fail."""
        stores = _make_valid_stores(n=5)
        txn = _make_valid_transactions(n=5)
        txn.loc[0, "store_id"] = "S999"  # not in stores
        result = validator.validate_transactions(txn, stores=stores)
        assert not result.passed
        check_names = [c["check"] for c in result.failed_checks]
        assert "store_id_referential_integrity" in check_names

    def test_all_valid_store_ids_pass(self, validator: DataValidator) -> None:
        """Transactions with store_ids all in stores table must pass ref-integrity."""
        stores = _make_valid_stores(n=5)
        txn = _make_valid_transactions(n=5)
        result = validator.validate_transactions(txn, stores=stores)
        assert result.passed, f"Failed checks: {result.failed_checks}"


# ── validate_all ──────────────────────────────────────────────────────────────


class TestValidateAll:
    def test_validate_all_returns_all_tables(self, validator: DataValidator) -> None:
        """validate_all must return results for every table in the input dict."""
        data = {
            "stores": _make_valid_stores(),
            "customers": _make_valid_customers(),
            "products": _make_valid_products(),
            "transactions": _make_valid_transactions(),
        }
        results = validator.validate_all(data)
        for table_name in data:
            assert table_name in results

    def test_validate_all_result_types(self, validator: DataValidator) -> None:
        """validate_all must return ValidationResult objects."""
        data = {
            "stores": _make_valid_stores(),
            "customers": _make_valid_customers(),
        }
        results = validator.validate_all(data)
        for _, result in results.items():
            assert isinstance(result, ValidationResult)

    def test_validate_all_valid_data_passes(self, validator: DataValidator) -> None:
        """All valid tables must pass when supplied together."""
        data = {
            "stores": _make_valid_stores(),
            "customers": _make_valid_customers(),
            "products": _make_valid_products(),
            "transactions": _make_valid_transactions(),
        }
        results = validator.validate_all(data)
        for table, result in results.items():
            assert result.passed, (
                f"Table '{table}' failed validation: {result.failed_checks}"
            )
