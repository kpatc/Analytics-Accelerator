"""Executive Overview — NovaMart Financial Performance."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Executive Overview | NovaMart", layout="wide", page_icon="📈")

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from dashboard.components.styles import inject_css, kpi_card, section_header, insight, page_header

inject_css()

page_header("Executive Overview", "NovaMart Financial Performance · 36-Month Analysis")

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def _load():
    from bcgx.data.loader import DataLoader
    loader = DataLoader()
    return loader.load_all()

try:
    data = _load()
    tx = data["transactions"]
    stores = data["stores"]
    tx["ym"] = pd.to_datetime(tx["date"]).dt.to_period("M")
except Exception as e:
    st.error(f"Data unavailable — run `make generate-data` first.\n{e}")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_rev    = tx["gross_revenue"].sum()
total_profit = tx["gross_profit"].sum()
avg_margin   = total_profit / total_rev * 100
months       = sorted(tx["ym"].unique())
early = tx[tx["ym"].isin(months[:12])]
late  = tx[tx["ym"].isin(months[-12:])]
m_start = early["gross_profit"].sum() / early["gross_revenue"].sum() * 100
m_end   = late["gross_profit"].sum()  / late["gross_revenue"].sum()  * 100
margin_drift = m_end - m_start
rev_yoy = (late["gross_revenue"].sum() - early["gross_revenue"].sum()) / early["gross_revenue"].sum() * 100

section_header("Financial Snapshot")
c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Total Revenue", f"${total_rev/1e6:.1f}M", sub="36 months")
with c2: kpi_card("Gross Profit", f"${total_profit/1e6:.1f}M", sub="36 months")
with c3: kpi_card("Avg Gross Margin", f"{avg_margin:.1f}%",
                   delta=f"{margin_drift:+.1f}pp Y1→Y3", positive=(margin_drift >= 0))
with c4: kpi_card("Revenue YoY", f"{rev_yoy:+.1f}%",
                   delta=f"Y1 → Y3", positive=(rev_yoy >= 0))

# ── Charts ────────────────────────────────────────────────────────────────────
section_header("Revenue & Margin Trends")
from dashboard.components.charts import revenue_trend_chart, margin_trend_chart, store_scatter_chart

col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(revenue_trend_chart(tx), use_container_width=True)
with col_r:
    st.plotly_chart(margin_trend_chart(tx), use_container_width=True)

# ── Store portfolio ────────────────────────────────────────────────────────────
section_header("Store Portfolio")
st.plotly_chart(store_scatter_chart(stores, tx), use_container_width=True)

# ── Cluster breakdown table ───────────────────────────────────────────────────
merged = tx.merge(stores[["store_id", "performance_cluster"]], on="store_id")
cluster_stats = (
    merged.groupby("performance_cluster")
    .agg(revenue=("gross_revenue", "sum"), profit=("gross_profit", "sum"), txns=("gross_revenue", "count"))
    .assign(
        margin=lambda d: d["profit"] / d["revenue"] * 100,
        rev_share=lambda d: d["revenue"] / d["revenue"].sum() * 100,
        profit_share=lambda d: d["profit"] / d["profit"].sum() * 100,
    )
    .reset_index()
)
n_by_cluster = stores.groupby("performance_cluster").size().reset_index(name="stores")
cluster_stats = cluster_stats.merge(n_by_cluster, on="performance_cluster")

col_a, col_b = st.columns([3, 2])
with col_a:
    display = cluster_stats[[
        "performance_cluster", "stores", "revenue", "profit", "margin", "rev_share", "profit_share"
    ]].copy()
    display.columns = ["Cluster", "Stores", "Revenue", "Profit", "Margin %", "Rev Share %", "Profit Share %"]
    display["Revenue"]      = display["Revenue"].apply(lambda v: f"${v/1e6:.1f}M")
    display["Profit"]       = display["Profit"].apply(lambda v: f"${v/1e6:.1f}M")
    display["Margin %"]     = display["Margin %"].apply(lambda v: f"{v:.1f}%")
    display["Rev Share %"]  = display["Rev Share %"].apply(lambda v: f"{v:.1f}%")
    display["Profit Share %"] = display["Profit Share %"].apply(lambda v: f"{v:.1f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)

with col_b:
    import plotly.graph_objects as go
    from dashboard.components.charts import BCGX_GREEN, BCGX_AMBER, BCGX_RED, CARD, TEXT, MUTED
    fig_bar = go.Figure(go.Bar(
        x=cluster_stats["performance_cluster"],
        y=cluster_stats["profit_share"],
        marker_color=[BCGX_GREEN, BCGX_AMBER, BCGX_RED],
        text=cluster_stats["profit_share"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(color=TEXT, size=12),
    ))
    fig_bar.update_layout(
        title="Profit Share by Cluster",
        yaxis_title="% of total profit",
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=MUTED, size=11),
        margin=dict(l=40, r=20, t=50, b=30),
        title_font=dict(size=13, color=TEXT),
        xaxis=dict(gridcolor="#21262D", linecolor="#30363D", tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="#21262D", linecolor="#30363D", tickfont=dict(color=MUTED)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
a_share = float(cluster_stats.loc[cluster_stats["performance_cluster"] == "A", "profit_share"].iloc[0])
c_share = float(cluster_stats.loc[cluster_stats["performance_cluster"] == "C", "profit_share"].iloc[0])
n_a = int(n_by_cluster.loc[n_by_cluster["performance_cluster"] == "A", "stores"].iloc[0])
n_c = int(n_by_cluster.loc[n_by_cluster["performance_cluster"] == "C", "stores"].iloc[0])

insight(
    f"Gross margin declined from <strong>{m_start:.1f}%</strong> to <strong>{m_end:.1f}%</strong> "
    f"({margin_drift:+.1f}pp) — driven by COGS inflation beginning Month 13 with no corresponding "
    f"price adjustment. Cluster A ({n_a} stores, {n_a/len(stores):.0%} of portfolio) generates "
    f"<strong>{a_share:.0f}%</strong> of profit while Cluster C ({n_c} stores) contributes only "
    f"<strong>{c_share:.0f}%</strong>."
)
