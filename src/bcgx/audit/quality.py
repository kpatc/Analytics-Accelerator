"""Data quality audit module for NovaMart datasets.

Profiles each DataFrame column (nulls, uniques, numeric statistics, outliers)
and checks business-level rules. Results are returned as structured dataclasses
suitable for serialisation and rich reporting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


# ── Profile dataclasses ───────────────────────────────────────────────────────


@dataclass
class ColumnProfile:
    """Statistical profile for a single DataFrame column."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    # Numeric stats (None for non-numeric columns)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None
    outlier_count: int | None = None  # IQR-based
    outlier_pct: float | None = None


@dataclass
class TableAudit:
    """Audit results for a single DataFrame."""

    table_name: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_pct: float
    columns: list[ColumnProfile] = field(default_factory=list)
    business_rule_violations: list[str] = field(default_factory=list)


# ── Auditor ───────────────────────────────────────────────────────────────────


class DataAuditor:
    """Profiles DataFrames and checks business-level quality rules.

    Usage::

        auditor = DataAuditor()
        audit = auditor.audit_table(df, "transactions")
        audits = auditor.audit_all(data_dict)
    """

    # Business rules per table: list of (rule_name, lambda: violation_msg | None)
    # Lambdas receive the DataFrame and return None if OK or a string if violated.

    _BUSINESS_RULES: dict[str, list[tuple[str, object]]] = {
        "transactions": [
            (
                "gross_profit = gross_revenue - cogs",
                lambda df: (
                    None
                    if "gross_revenue" not in df.columns
                    or "cogs" not in df.columns
                    or "gross_profit" not in df.columns
                    else (
                        None
                        if (
                            np.abs(
                                df["gross_profit"] - (df["gross_revenue"] - df["cogs"])
                            ).max()
                            <= 0.02
                        )
                        else f"{(np.abs(df['gross_profit'] - (df['gross_revenue'] - df['cogs'])) > 0.02).sum()} rows violate gross_profit = gross_revenue - cogs"
                    )
                ),
            ),
            (
                "unit_price > 0",
                lambda df: (
                    None
                    if "unit_price" not in df.columns
                    else (
                        None
                        if (df["unit_price"] > 0).all()
                        else f"{(df['unit_price'] <= 0).sum()} rows with unit_price <= 0"
                    )
                ),
            ),
            (
                "quantity > 0",
                lambda df: (
                    None
                    if "quantity" not in df.columns
                    else (
                        None
                        if (df["quantity"] > 0).all()
                        else f"{(df['quantity'] <= 0).sum()} rows with quantity <= 0"
                    )
                ),
            ),
        ],
        "stores": [
            (
                "cluster_profit_multiplier matches cluster",
                lambda df: (
                    None
                    if "performance_cluster" not in df.columns
                    or "cluster_profit_multiplier" not in df.columns
                    else _check_cluster_multipliers(df)
                ),
            ),
            (
                "sq_footage > 0",
                lambda df: (
                    None
                    if "sq_footage" not in df.columns
                    else (
                        None
                        if (df["sq_footage"] > 0).all()
                        else f"{(df['sq_footage'] <= 0).sum()} stores with sq_footage <= 0"
                    )
                ),
            ),
        ],
        "products": [
            (
                "list_price > unit_cost",
                lambda df: (
                    None
                    if "list_price" not in df.columns or "unit_cost" not in df.columns
                    else (
                        None
                        if (df["list_price"] >= df["unit_cost"]).all()
                        else f"{(df['list_price'] < df['unit_cost']).sum()} products with list_price < unit_cost"
                    )
                ),
            ),
            (
                "private_label margin > national_brand margin",
                lambda df: _check_brand_margin(df),
            ),
        ],
        "marketing_spend": [
            (
                "spend_usd >= 0",
                lambda df: (
                    None
                    if "spend_usd" not in df.columns
                    else (
                        None
                        if (df["spend_usd"] >= 0).all()
                        else f"{(df['spend_usd'] < 0).sum()} rows with negative spend_usd"
                    )
                ),
            ),
        ],
        "store_costs": [
            (
                "total_operating_cost_usd > 0",
                lambda df: (
                    None
                    if "total_operating_cost_usd" not in df.columns
                    else (
                        None
                        if (df["total_operating_cost_usd"] > 0).all()
                        else f"{(df['total_operating_cost_usd'] <= 0).sum()} rows with total_operating_cost <= 0"
                    )
                ),
            ),
        ],
    }

    def audit_table(self, df: pd.DataFrame, table_name: str) -> TableAudit:
        """Profile a single DataFrame and check business rules.

        Args:
            df: DataFrame to audit.
            table_name: Logical name of the table (used for rule lookup).

        Returns:
            TableAudit with column profiles and rule violations.
        """
        logger.info(f"Auditing table '{table_name}' ({df.shape[0]:,} rows × {df.shape[1]} cols)")

        row_count = len(df)
        col_count = len(df.columns)
        dup_count = int(df.duplicated().sum())
        dup_pct = round(dup_count / row_count * 100, 4) if row_count > 0 else 0.0

        columns: list[ColumnProfile] = []
        for col in df.columns:
            columns.append(self._profile_column(df[col], row_count))

        # Business rule checks
        violations: list[str] = []
        for rule_name, rule_fn in self._BUSINESS_RULES.get(table_name, []):
            try:
                result = rule_fn(df)  # type: ignore[operator]
                if result is not None:
                    violations.append(f"{rule_name}: {result}")
            except Exception as exc:
                violations.append(f"{rule_name}: ERROR during check — {exc}")

        audit = TableAudit(
            table_name=table_name,
            row_count=row_count,
            column_count=col_count,
            duplicate_row_count=dup_count,
            duplicate_row_pct=dup_pct,
            columns=columns,
            business_rule_violations=violations,
        )

        if violations:
            logger.warning(f"  {len(violations)} business rule violation(s) in '{table_name}'")
        else:
            logger.success(f"  No business rule violations in '{table_name}'")

        return audit

    def audit_all(self, data: dict[str, pd.DataFrame]) -> dict[str, TableAudit]:
        """Audit every DataFrame in the data dictionary.

        Args:
            data: Dict mapping table names to DataFrames.

        Returns:
            Dict mapping table names to TableAudit results.
        """
        audits: dict[str, TableAudit] = {}
        for name, df in data.items():
            audits[name] = self.audit_table(df, name)
        logger.success(f"Audit complete: {len(audits)} tables profiled")
        return audits

    def to_dict(self, audit: TableAudit) -> dict:
        """Serialise a TableAudit to a plain Python dict (JSON-safe).

        Args:
            audit: TableAudit to serialise.

        Returns:
            Dict representation suitable for json.dumps.
        """
        return asdict(audit)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _profile_column(series: pd.Series, total_rows: int) -> ColumnProfile:
        """Compute a ColumnProfile for a single pandas Series."""
        name = str(series.name)
        dtype = str(series.dtype)
        null_count = int(series.isna().sum())
        null_pct = round(null_count / total_rows * 100, 4) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_pct = round(unique_count / total_rows * 100, 4) if total_rows > 0 else 0.0

        # Numeric stats
        mean = std = mn = p25 = p50 = p75 = mx = None
        outlier_count = outlier_pct = None

        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) > 0 and pd.api.types.is_numeric_dtype(series):
            mean = float(round(numeric_series.mean(), 6))
            std = float(round(numeric_series.std(), 6))
            mn = float(round(numeric_series.min(), 6))
            p25 = float(round(numeric_series.quantile(0.25), 6))
            p50 = float(round(numeric_series.quantile(0.50), 6))
            p75 = float(round(numeric_series.quantile(0.75), 6))
            mx = float(round(numeric_series.max(), 6))

            # IQR-based outlier detection
            iqr = p75 - p25
            lower_fence = p25 - 1.5 * iqr
            upper_fence = p75 + 1.5 * iqr
            outliers = ((numeric_series < lower_fence) | (numeric_series > upper_fence))
            outlier_count = int(outliers.sum())
            outlier_pct = round(outlier_count / len(numeric_series) * 100, 4)

        return ColumnProfile(
            name=name,
            dtype=dtype,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            mean=mean,
            std=std,
            min=mn,
            p25=p25,
            p50=p50,
            p75=p75,
            max=mx,
            outlier_count=outlier_count,
            outlier_pct=outlier_pct,
        )


# ── Module-level helper functions (used in business rules) ────────────────────


def _check_cluster_multipliers(df: pd.DataFrame) -> str | None:
    """Check that cluster_profit_multiplier matches the expected values for each cluster."""
    expected = {"A": 2.8, "B": 1.0, "C": 0.4}
    violations = 0
    for cluster, expected_mult in expected.items():
        cluster_df = df[df["performance_cluster"] == cluster]
        if len(cluster_df) > 0:
            wrong = (cluster_df["cluster_profit_multiplier"] != expected_mult).sum()
            violations += int(wrong)
    if violations:
        return f"{violations} rows with unexpected cluster_profit_multiplier"
    return None


def _check_brand_margin(df: pd.DataFrame) -> str | None:
    """Check that private_label margin > national_brand margin."""
    required = {"brand_type", "unit_cost", "list_price"}
    if not required.issubset(df.columns):
        return None
    pl = df[df["brand_type"] == "private_label"]
    nb = df[df["brand_type"] == "national_brand"]
    if len(pl) == 0 or len(nb) == 0:
        return None
    pl_margin = ((pl["list_price"] - pl["unit_cost"]) / pl["list_price"]).mean()
    nb_margin = ((nb["list_price"] - nb["unit_cost"]) / nb["list_price"]).mean()
    if pl_margin <= nb_margin:
        return f"Private label avg margin ({pl_margin:.3f}) ≤ national brand ({nb_margin:.3f})"
    return None
