"""Data validation module for NovaMart synthetic datasets.

Validates each DataFrame against schema requirements, referential integrity rules,
and business logic constraints. Returns structured ValidationResult objects that
record which checks passed/failed and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Outcome of validating a single table."""

    table: str
    passed: bool
    checks: list[dict]  # [{"check": str, "passed": bool, "detail": str}]

    @property
    def failed_checks(self) -> list[dict]:
        """Return only the checks that failed."""
        return [c for c in self.checks if not c["passed"]]

    @property
    def n_passed(self) -> int:
        """Count of checks that passed."""
        return sum(1 for c in self.checks if c["passed"])

    @property
    def n_failed(self) -> int:
        """Count of checks that failed."""
        return len(self.checks) - self.n_passed


# ── Helpers ───────────────────────────────────────────────────────────────────


def _check(name: str, condition: bool, detail: str) -> dict:
    """Build a single check result dict."""
    return {"check": name, "passed": condition, "detail": detail}


def _result(table: str, checks: list[dict]) -> ValidationResult:
    """Build a ValidationResult from a list of checks."""
    passed = all(c["passed"] for c in checks)
    return ValidationResult(table=table, passed=passed, checks=checks)


# ── Validator ────────────────────────────────────────────────────────────────


class DataValidator:
    """Validates NovaMart DataFrames against schema and business rules."""

    # ── Stores ────────────────────────────────────────────────────────────

    def validate_stores(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the stores DataFrame.

        Checks: schema columns, no null PKs, no duplicate PKs, numeric ranges,
        categorical value sets, and cluster distribution.

        Args:
            df: Stores DataFrame to validate.

        Returns:
            ValidationResult with all check outcomes.
        """
        checks: list[dict] = []
        required_cols = [
            "store_id", "region", "store_format", "sq_footage",
            "open_date", "manager_tenure_years", "performance_cluster",
            "cluster_profit_multiplier",
        ]

        # Schema check
        missing_cols = [c for c in required_cols if c not in df.columns]
        checks.append(_check(
            "required_columns_present",
            len(missing_cols) == 0,
            f"Missing columns: {missing_cols}" if missing_cols else "All required columns present",
        ))

        if "store_id" in df.columns:
            # No null PKs
            null_pks = df["store_id"].isna().sum()
            checks.append(_check(
                "no_null_store_id",
                null_pks == 0,
                f"{null_pks} null store_id values" if null_pks else "No null PKs",
            ))

            # No duplicate PKs
            dup_pks = df["store_id"].duplicated().sum()
            checks.append(_check(
                "no_duplicate_store_id",
                dup_pks == 0,
                f"{dup_pks} duplicate store_id values" if dup_pks else "No duplicate PKs",
            ))

        if "store_format" in df.columns:
            valid_formats = {"urban", "suburban", "rural"}
            invalid_formats = set(df["store_format"].dropna().unique()) - valid_formats
            checks.append(_check(
                "valid_store_format",
                len(invalid_formats) == 0,
                f"Invalid formats: {invalid_formats}" if invalid_formats else "All formats valid",
            ))

        if "region" in df.columns:
            valid_regions = {"Northeast", "Southeast", "Midwest", "Southwest", "West"}
            invalid_regions = set(df["region"].dropna().unique()) - valid_regions
            checks.append(_check(
                "valid_region",
                len(invalid_regions) == 0,
                f"Invalid regions: {invalid_regions}" if invalid_regions else "All regions valid",
            ))

        if "sq_footage" in df.columns:
            neg_sqft = (df["sq_footage"] <= 0).sum()
            checks.append(_check(
                "sq_footage_positive",
                neg_sqft == 0,
                f"{neg_sqft} non-positive sq_footage values" if neg_sqft else "All sq_footage > 0",
            ))

        if "manager_tenure_years" in df.columns:
            out_of_range = ((df["manager_tenure_years"] < 0) | (df["manager_tenure_years"] > 50)).sum()
            checks.append(_check(
                "manager_tenure_range",
                out_of_range == 0,
                f"{out_of_range} tenure values out of [0, 50]" if out_of_range else "Tenure in range",
            ))

        if "performance_cluster" in df.columns:
            valid_clusters = {"A", "B", "C"}
            invalid_clusters = set(df["performance_cluster"].dropna().unique()) - valid_clusters
            checks.append(_check(
                "valid_performance_cluster",
                len(invalid_clusters) == 0,
                f"Invalid clusters: {invalid_clusters}" if invalid_clusters else "All clusters valid",
            ))

        if "cluster_profit_multiplier" in df.columns:
            neg_mult = (df["cluster_profit_multiplier"] <= 0).sum()
            checks.append(_check(
                "cluster_profit_multiplier_positive",
                neg_mult == 0,
                f"{neg_mult} non-positive multipliers" if neg_mult else "All multipliers > 0",
            ))

        result = _result("stores", checks)
        self._log_result(result)
        return result

    # ── Customers ─────────────────────────────────────────────────────────

    def validate_customers(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the customers DataFrame.

        Args:
            df: Customers DataFrame to validate.

        Returns:
            ValidationResult with all check outcomes.
        """
        checks: list[dict] = []
        required_cols = [
            "customer_id", "segment", "loyalty_tier", "acquisition_channel",
            "region", "acquisition_date", "email_opt_in", "age_group",
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]
        checks.append(_check(
            "required_columns_present",
            len(missing_cols) == 0,
            f"Missing columns: {missing_cols}" if missing_cols else "All required columns present",
        ))

        if "customer_id" in df.columns:
            null_pks = df["customer_id"].isna().sum()
            checks.append(_check(
                "no_null_customer_id",
                null_pks == 0,
                f"{null_pks} null customer_id values" if null_pks else "No null PKs",
            ))

            dup_pks = df["customer_id"].duplicated().sum()
            checks.append(_check(
                "no_duplicate_customer_id",
                dup_pks == 0,
                f"{dup_pks} duplicate customer_id values" if dup_pks else "No duplicate PKs",
            ))

        if "loyalty_tier" in df.columns:
            valid_tiers = {"Gold", "Silver", "Bronze"}
            invalid_tiers = set(df["loyalty_tier"].dropna().unique()) - valid_tiers
            checks.append(_check(
                "valid_loyalty_tier",
                len(invalid_tiers) == 0,
                f"Invalid tiers: {invalid_tiers}" if invalid_tiers else "All tiers valid",
            ))

        if "segment" in df.columns:
            valid_segments = {"Premium", "Regular", "Occasional"}
            invalid_segs = set(df["segment"].dropna().unique()) - valid_segments
            checks.append(_check(
                "valid_segment",
                len(invalid_segs) == 0,
                f"Invalid segments: {invalid_segs}" if invalid_segs else "All segments valid",
            ))

        if "email_opt_in" in df.columns:
            non_bool = (~df["email_opt_in"].isin([True, False])).sum()
            checks.append(_check(
                "email_opt_in_boolean",
                non_bool == 0,
                f"{non_bool} non-boolean email_opt_in values" if non_bool else "All boolean",
            ))

        result = _result("customers", checks)
        self._log_result(result)
        return result

    # ── Products ──────────────────────────────────────────────────────────

    def validate_products(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the products DataFrame.

        Checks prices are positive, private label margin > national brand margin.

        Args:
            df: Products DataFrame to validate.

        Returns:
            ValidationResult with all check outcomes.
        """
        checks: list[dict] = []
        required_cols = [
            "product_id", "category", "subcategory", "brand_type",
            "unit_cost", "list_price", "price_elasticity_coefficient",
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]
        checks.append(_check(
            "required_columns_present",
            len(missing_cols) == 0,
            f"Missing columns: {missing_cols}" if missing_cols else "All required columns present",
        ))

        if "product_id" in df.columns:
            null_pks = df["product_id"].isna().sum()
            checks.append(_check(
                "no_null_product_id",
                null_pks == 0,
                f"{null_pks} null product_id values" if null_pks else "No null PKs",
            ))

            dup_pks = df["product_id"].duplicated().sum()
            checks.append(_check(
                "no_duplicate_product_id",
                dup_pks == 0,
                f"{dup_pks} duplicate product_id values" if dup_pks else "No duplicate PKs",
            ))

        if "list_price" in df.columns:
            neg_price = (df["list_price"] <= 0).sum()
            checks.append(_check(
                "list_price_positive",
                neg_price == 0,
                f"{neg_price} non-positive list_price values" if neg_price else "All list prices > 0",
            ))

        if "unit_cost" in df.columns:
            neg_cost = (df["unit_cost"] <= 0).sum()
            checks.append(_check(
                "unit_cost_positive",
                neg_cost == 0,
                f"{neg_cost} non-positive unit_cost values" if neg_cost else "All unit costs > 0",
            ))

        if all(c in df.columns for c in ["unit_cost", "list_price"]):
            invalid_margin = (df["list_price"] < df["unit_cost"]).sum()
            checks.append(_check(
                "list_price_exceeds_cost",
                invalid_margin == 0,
                f"{invalid_margin} products where list_price < unit_cost" if invalid_margin else "All margins positive",
            ))

        if all(c in df.columns for c in ["brand_type", "unit_cost", "list_price"]):
            pl = df[df["brand_type"] == "private_label"]
            nb = df[df["brand_type"] == "national_brand"]
            if len(pl) > 0 and len(nb) > 0:
                pl_margin = ((pl["list_price"] - pl["unit_cost"]) / pl["list_price"]).mean()
                nb_margin = ((nb["list_price"] - nb["unit_cost"]) / nb["list_price"]).mean()
                checks.append(_check(
                    "private_label_higher_margin",
                    pl_margin > nb_margin,
                    f"PL margin={pl_margin:.3f} NB margin={nb_margin:.3f}",
                ))

        if "brand_type" in df.columns:
            valid_brand_types = {"private_label", "national_brand"}
            invalid_brands = set(df["brand_type"].dropna().unique()) - valid_brand_types
            checks.append(_check(
                "valid_brand_type",
                len(invalid_brands) == 0,
                f"Invalid brand types: {invalid_brands}" if invalid_brands else "All brand types valid",
            ))

        if "category" in df.columns:
            valid_categories = {
                "Electronics", "Apparel", "Home & Garden", "Food & Beverage",
                "Health & Beauty", "Sports", "Toys",
            }
            invalid_cats = set(df["category"].dropna().unique()) - valid_categories
            checks.append(_check(
                "valid_category",
                len(invalid_cats) == 0,
                f"Invalid categories: {invalid_cats}" if invalid_cats else "All categories valid",
            ))

        result = _result("products", checks)
        self._log_result(result)
        return result

    # ── Transactions ──────────────────────────────────────────────────────

    def validate_transactions(
        self,
        df: pd.DataFrame,
        stores: pd.DataFrame | None = None,
        customers: pd.DataFrame | None = None,
        products: pd.DataFrame | None = None,
    ) -> ValidationResult:
        """Validate the transactions DataFrame.

        Optionally validates referential integrity against stores/customers/products.

        Args:
            df: Transactions DataFrame to validate.
            stores: Store master DataFrame for referential integrity (optional).
            customers: Customer master DataFrame for referential integrity (optional).
            products: Product catalogue DataFrame for referential integrity (optional).

        Returns:
            ValidationResult with all check outcomes.
        """
        checks: list[dict] = []
        required_cols = [
            "transaction_id", "date", "store_id", "customer_id", "product_id",
            "quantity", "unit_price", "gross_revenue", "cogs", "gross_profit",
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]
        checks.append(_check(
            "required_columns_present",
            len(missing_cols) == 0,
            f"Missing columns: {missing_cols}" if missing_cols else "All required columns present",
        ))

        if "transaction_id" in df.columns:
            null_pks = df["transaction_id"].isna().sum()
            checks.append(_check(
                "no_null_transaction_id",
                null_pks == 0,
                f"{null_pks} null transaction_id values" if null_pks else "No null PKs",
            ))

            dup_pks = df["transaction_id"].duplicated().sum()
            checks.append(_check(
                "no_duplicate_transaction_id",
                dup_pks == 0,
                f"{dup_pks} duplicate transaction_id values" if dup_pks else "No duplicate PKs",
            ))

        if "quantity" in df.columns:
            non_pos_qty = (df["quantity"] <= 0).sum()
            checks.append(_check(
                "quantity_positive",
                non_pos_qty == 0,
                f"{non_pos_qty} non-positive quantity values" if non_pos_qty else "All quantities > 0",
            ))

        if "unit_price" in df.columns:
            neg_price = (df["unit_price"] <= 0).sum()
            checks.append(_check(
                "unit_price_positive",
                neg_price == 0,
                f"{neg_price} non-positive unit_price values" if neg_price else "All unit prices > 0",
            ))

        if "gross_revenue" in df.columns:
            neg_rev = (df["gross_revenue"] < 0).sum()
            checks.append(_check(
                "gross_revenue_non_negative",
                neg_rev == 0,
                f"{neg_rev} negative gross_revenue values" if neg_rev else "All gross_revenue >= 0",
            ))

        if "cogs" in df.columns:
            neg_cogs = (df["cogs"] < 0).sum()
            checks.append(_check(
                "cogs_non_negative",
                neg_cogs == 0,
                f"{neg_cogs} negative cogs values" if neg_cogs else "All cogs >= 0",
            ))

        # Business rule: gross_profit = gross_revenue - cogs (within $0.02 tolerance)
        if all(c in df.columns for c in ["gross_revenue", "cogs", "gross_profit"]):
            computed = (df["gross_revenue"] - df["cogs"]).round(2)
            tolerance = 0.02
            violations = (np.abs(df["gross_profit"] - computed) > tolerance).sum()
            checks.append(_check(
                "gross_profit_equals_revenue_minus_cogs",
                violations == 0,
                f"{violations} rows where gross_profit ≠ gross_revenue - cogs (tol={tolerance})"
                if violations else "gross_profit = gross_revenue - cogs (within tolerance)",
            ))

        if "discount_pct" in df.columns:
            invalid_discount = ((df["discount_pct"] < 0) | (df["discount_pct"] > 1)).sum()
            checks.append(_check(
                "discount_pct_range",
                invalid_discount == 0,
                f"{invalid_discount} discount_pct values outside [0, 1]"
                if invalid_discount else "All discount_pct in [0, 1]",
            ))

        # Referential integrity
        if stores is not None and "store_id" in df.columns and "store_id" in stores.columns:
            valid_store_ids = set(stores["store_id"].tolist())
            orphan_stores = (~df["store_id"].isin(valid_store_ids)).sum()
            checks.append(_check(
                "store_id_referential_integrity",
                orphan_stores == 0,
                f"{orphan_stores} transaction store_ids not in stores table"
                if orphan_stores else "All store_ids valid",
            ))

        if customers is not None and "customer_id" in df.columns and "customer_id" in customers.columns:
            valid_customer_ids = set(customers["customer_id"].tolist())
            orphan_customers = (~df["customer_id"].isin(valid_customer_ids)).sum()
            checks.append(_check(
                "customer_id_referential_integrity",
                orphan_customers == 0,
                f"{orphan_customers} transaction customer_ids not in customers table"
                if orphan_customers else "All customer_ids valid",
            ))

        if products is not None and "product_id" in df.columns and "product_id" in products.columns:
            valid_product_ids = set(products["product_id"].tolist())
            orphan_products = (~df["product_id"].isin(valid_product_ids)).sum()
            checks.append(_check(
                "product_id_referential_integrity",
                orphan_products == 0,
                f"{orphan_products} transaction product_ids not in products table"
                if orphan_products else "All product_ids valid",
            ))

        result = _result("transactions", checks)
        self._log_result(result)
        return result

    # ── Aggregate ─────────────────────────────────────────────────────────

    def validate_all(self, data: dict[str, pd.DataFrame]) -> dict[str, ValidationResult]:
        """Run all available validations against the provided data dictionary.

        Expects keys: 'stores', 'customers', 'products', 'transactions'.
        Referential integrity checks are run when all referenced tables are present.

        Args:
            data: Dictionary mapping table names to DataFrames.

        Returns:
            Dictionary mapping table names to ValidationResult objects.
        """
        results: dict[str, ValidationResult] = {}

        if "stores" in data:
            results["stores"] = self.validate_stores(data["stores"])

        if "customers" in data:
            results["customers"] = self.validate_customers(data["customers"])

        if "products" in data:
            results["products"] = self.validate_products(data["products"])

        if "transactions" in data:
            results["transactions"] = self.validate_transactions(
                df=data["transactions"],
                stores=data.get("stores"),
                customers=data.get("customers"),
                products=data.get("products"),
            )

        n_total = sum(len(r.checks) for r in results.values())
        n_failed = sum(r.n_failed for r in results.values())
        logger.info(f"Validation complete: {n_total} checks | {n_failed} failed")

        return results

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _log_result(result: ValidationResult) -> None:
        """Log the outcome of a validation result."""
        if result.passed:
            logger.success(
                f"[{result.table}] Validation PASSED ({result.n_passed}/{len(result.checks)} checks)"
            )
        else:
            logger.warning(
                f"[{result.table}] Validation FAILED ({result.n_failed} checks failed)"
            )
            for check in result.failed_checks:
                logger.warning(f"  ✗ {check['check']}: {check['detail']}")
