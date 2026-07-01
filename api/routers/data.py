"""Data layer router.

Endpoints that expose high-level KPIs and dataset metadata for the executive dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Ensure src is importable when running via uvicorn from repo root
_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

router = APIRouter()


@router.get(
    "/data/summary",
    summary="Executive KPI summary",
    description="Returns high-level KPIs for the executive dashboard — total revenue, profit, margin, etc.",
)
async def get_data_summary() -> dict:
    """Return high-level KPIs computed from the NovaMart transaction dataset."""
    try:
        from bcgx.data.loader import DataLoader

        loader = DataLoader()
        data = loader.load_all()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Data not available: {exc}. Run 'python scripts/generate_data.py' first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data layer error: {exc}")

    tx = data["transactions"]
    stores = data["stores"]
    customers = data["customers"]

    total_revenue = float(tx["gross_revenue"].sum())
    total_profit = float(tx["gross_profit"].sum())
    avg_gross_margin_pct = float(total_profit / total_revenue * 100) if total_revenue > 0 else 0.0

    return {
        "total_revenue_usd": round(total_revenue, 2),
        "total_profit_usd": round(total_profit, 2),
        "avg_gross_margin_pct": round(avg_gross_margin_pct, 4),
        "total_transactions": int(len(tx)),
        "total_stores": int(len(stores)),
        "total_customers": int(len(customers)),
        "date_range_start": str(tx["date"].min().date()),
        "date_range_end": str(tx["date"].max().date()),
        "n_months": int(tx["date"].dt.to_period("M").nunique()),
        "store_formats": sorted(stores["store_format"].unique().tolist()),
        "product_categories": sorted(
            data["products"]["category"].unique().tolist()
        ),
        "loyalty_tiers": sorted(customers["loyalty_tier"].unique().tolist()),
    }


@router.get(
    "/data/datasets",
    summary="Dataset metadata",
    description="Returns row counts and column counts for all available datasets.",
)
async def get_dataset_metadata() -> dict:
    """Return metadata about all available NovaMart datasets."""
    try:
        from bcgx.data.loader import DataLoader

        loader = DataLoader()
        data = loader.load_all()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Data not available: {exc}. Run 'python scripts/generate_data.py' first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data layer error: {exc}")

    return {
        table: {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }
        for table, df in data.items()
    }
