"""Executive Overview — NovaMart Financial Performance."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(page_title="Executive Overview | NovaMart", layout="wide")

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

st.title("Executive Overview — NovaMart Financial Performance")
st.caption("BCG X Analytics Accelerator | Confidential")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    from bcgx.data.loader import DataLoader

    loader = DataLoader()
    data = loader.load_all()
    tx = data["transactions"]
    stores = data["stores"]
    customers = data["customers"]
    DATA_OK = True
except Exception as e:
    st.error(f"Data not available: {e}\nRun `python scripts/generate_data.py` first.")
    DATA_OK = False

if not DATA_OK:
    st.stop()

# ── Compute KPIs ───────────────────────────────────────────────────────────────
total_rev = tx["gross_revenue"].sum()
total_profit = tx["gross_profit"].sum()
avg_margin = total_profit / total_rev * 100
n_months = tx["date"].dt.to_period("M").nunique()

# YoY: compare first 12 months vs last 12 months
tx_sorted = tx.sort_values("date")
month_labels = sorted(tx["year_month"].unique())
first12 = set(month_labels[:12])
last12 = set(month_labels[-12:])
rev_y1 = tx[tx["year_month"].isin(first12)]["gross_revenue"].sum()
rev_y3 = tx[tx["year_month"].isin(last12)]["gross_revenue"].sum()
yoy_change = (rev_y3 - rev_y1) / rev_y1 * 100 if rev_y1 > 0 else 0.0

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue (36m)", f"${total_rev/1e6:.1f}M")
c2.metric("Total Gross Profit (36m)", f"${total_profit/1e6:.1f}M")
c3.metric("Avg Gross Margin", f"{avg_margin:.1f}%")
c4.metric(
    "Revenue YoY Change (Y1→Y3)",
    f"{yoy_change:+.1f}%",
    delta=f"{yoy_change:+.1f}%",
    delta_color="normal",
)

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
from dashboard.components.charts import (
    margin_trend_chart,
    revenue_trend_chart,
    store_scatter_chart,
)

col_left, col_right = st.columns(2)

with col_left:
    fig_rev = revenue_trend_chart(tx)
    st.plotly_chart(fig_rev, use_container_width=True)

with col_right:
    fig_margin = margin_trend_chart(tx)
    st.plotly_chart(fig_margin, use_container_width=True)

# ── Store scatter ──────────────────────────────────────────────────────────────
st.subheader("Store Performance Portfolio")
fig_stores = store_scatter_chart(stores, tx)
st.plotly_chart(fig_stores, use_container_width=True)

# ── Key Insight ───────────────────────────────────────────────────────────────
first_margin = (
    tx[tx["year_month"].isin(first12)]["gross_profit"].sum()
    / tx[tx["year_month"].isin(first12)]["gross_revenue"].sum()
    * 100
)
last_margin = (
    tx[tx["year_month"].isin(last12)]["gross_profit"].sum()
    / tx[tx["year_month"].isin(last12)]["gross_revenue"].sum()
    * 100
)
margin_decline = first_margin - last_margin

cluster_c = stores[stores["performance_cluster"] == "C"]
cluster_c_profit = tx[tx["store_id"].isin(cluster_c["store_id"])]["gross_profit"].sum()
cluster_c_share = cluster_c_profit / total_profit * 100

st.info(
    f"**Key Insight:** Gross margin declined from {first_margin:.1f}% in Year 1 to "
    f"{last_margin:.1f}% in Year 3 (-{margin_decline:.1f}pp). "
    f"Cluster C stores ({len(cluster_c)} locations, {len(cluster_c)/len(stores):.0%} of estate) "
    f"generate only {cluster_c_share:.1f}% of total profit — "
    f"significant portfolio concentration risk."
)
