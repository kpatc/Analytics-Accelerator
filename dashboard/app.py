"""NovaMart Analytics — Dashboard landing page."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="NovaMart Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from dashboard.components.styles import inject_css, kpi_card, section_header, insight

inject_css()

# ── Sidebar brand ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="padding:0.5rem 0 1.5rem 0; border-bottom:1px solid #30363D; margin-bottom:1rem;">
            <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;color:#00A651;text-transform:uppercase;">
                Analytics Simulation
            </div>
            <div style="font-size:1rem;font-weight:600;color:#E6EDF3;margin-top:0.2rem;">
                NovaMart Intelligence
            </div>
            <div style="font-size:0.7rem;color:#8B949E;margin-top:0.15rem;">
                BCG X–Inspired Methodology
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding:2rem 2rem 1.5rem 2rem; background:linear-gradient(135deg,#161B27 0%,#1C2333 100%);
         border:1px solid #30363D; border-radius:12px; margin-bottom:2rem;">
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;color:#00A651;
                    text-transform:uppercase;margin-bottom:0.6rem;">
            Simulated Engagement &nbsp;·&nbsp; Advanced Analytics &nbsp;·&nbsp; BCG X–Inspired Delivery
        </div>
        <h1 style="font-size:2rem;font-weight:700;color:#E6EDF3;margin:0 0 0.4rem 0;letter-spacing:-0.03em;">
            NovaMart Retail Intelligence
        </h1>
        <p style="color:#8B949E;font-size:0.9rem;margin:0;max-width:620px;line-height:1.6;">
            End-to-end analytics engagement covering margin diagnosis, customer churn,
            store portfolio optimization, pricing elasticity, and marketing ROI —
            36 months of transaction data across 800 stores.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
_data_ok = False
try:
    from bcgx.data.loader import DataLoader
    _loader = DataLoader()
    _data = _loader.load_all()
    _tx = _data["transactions"]
    _stores = _data["stores"]
    _customers = _data["customers"]
    _data_ok = True
except Exception:
    pass

# ── KPI row ───────────────────────────────────────────────────────────────────
section_header("Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)

if _data_ok:
    total_rev = _tx["gross_revenue"].sum()
    total_profit = _tx["gross_profit"].sum()
    margin = total_profit / total_rev * 100
    import pandas as pd
    _tx["ym"] = pd.to_datetime(_tx["date"]).dt.to_period("M")
    months = sorted(_tx["ym"].unique())
    m_early = _tx[_tx["ym"].isin(months[:3])]
    m_late  = _tx[_tx["ym"].isin(months[-3:])]
    m_start = m_early["gross_profit"].sum() / m_early["gross_revenue"].sum() * 100
    m_end   = m_late["gross_profit"].sum()  / m_late["gross_revenue"].sum()  * 100
    margin_drift = m_end - m_start

    with c1:
        kpi_card("Total Revenue", f"${total_rev/1e6:.1f}M", sub="36-month cumulative")
    with c2:
        kpi_card("Gross Profit", f"${total_profit/1e6:.1f}M", sub="36-month cumulative")
    with c3:
        kpi_card("Avg Gross Margin", f"{margin:.1f}%",
                 delta=f"{margin_drift:+.1f}pp Y1→Y3", positive=(margin_drift >= 0))
    with c4:
        kpi_card("Stores", f"{len(_stores):,}", sub="Active locations")
    with c5:
        kpi_card("Loyalty Customers", f"{len(_customers)/1e3:.0f}K", sub="Enrolled members")

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    insight(
        f"<strong>Data loaded.</strong> {len(_tx):,} transactions across "
        f"{_tx['ym'].nunique()} months. "
        f"Margin trend: <strong>{m_start:.1f}% → {m_end:.1f}%</strong> "
        f"({'▼' if margin_drift < 0 else '▲'} {abs(margin_drift):.1f}pp). "
        f"Navigate via the sidebar to explore each analysis module."
    )
else:
    with c1: kpi_card("Total Revenue", "—")
    with c2: kpi_card("Gross Profit", "—")
    with c3: kpi_card("Avg Gross Margin", "—")
    with c4: kpi_card("Stores", "—")
    with c5: kpi_card("Loyalty Customers", "—")
    st.markdown(
        '<div class="warn-box"><strong>No data found.</strong> Run <code>make generate-data</code> to populate the dashboard.</div>',
        unsafe_allow_html=True,
    )

# ── Module cards ──────────────────────────────────────────────────────────────
section_header("Analytics Modules")

modules = [
    ("📈", "Executive Overview", "Revenue, profit, margin trends and store portfolio heatmap."),
    ("🔍", "Data Audit", "Quality scorecard — completeness, duplicates, outlier detection."),
    ("📊", "Exploratory Analysis", "Univariate, bivariate and temporal business insights."),
    ("🧪", "Statistical Analysis", "Hypothesis tests, RFM segmentation, price elasticity."),
    ("🤖", "Model Performance", "Churn, store, MMM and elasticity model comparison + SHAP."),
    ("⚡", "Scenario Simulator", "What-if: price change, marketing reallocation, churn reduction."),
    ("🎯", "Recommendations", "8 RICE-scored strategic actions ranked by revenue impact."),
    ("💬", "AI Copilot", "Ask any business question — grounded answers from live data."),
]

cols = st.columns(4)
for i, (icon, name, desc) in enumerate(modules):
    with cols[i % 4]:
        st.markdown(
            f"""<div style="background:#1C2333;border:1px solid #30363D;border-radius:10px;
                    padding:1.4rem 1.5rem;margin-bottom:0.75rem;">
                <div style="font-size:1.6rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:0.95rem;font-weight:600;color:#E6EDF3;margin-bottom:0.35rem;">
                    {name}
                </div>
                <div style="font-size:0.82rem;color:#8B949E;line-height:1.55;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #21262D;
            display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:0.7rem;color:#8B949E;">
            Analytics Simulation &nbsp;·&nbsp; NovaMart Engagement &nbsp;·&nbsp; BCG X–Inspired
        </span>
        <span style="font-size:0.7rem;color:#8B949E;">v0.1.0</span>
    </div>""",
    unsafe_allow_html=True,
)
