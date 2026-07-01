"""BCG X NovaMart Analytics — Streamlit Dashboard Entry Point.

Landing page with KPI cards loaded from the data layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Page configuration (must be the first Streamlit call) ────────────────────
st.set_page_config(
    page_title="BCG X | NovaMart Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure src is importable
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

# ── Custom CSS — BCG X brand palette ─────────────────────────────────────────
st.markdown(
    """
    <style>
        :root {
            --bcgx-green:  #00A651;
            --bcgx-dark:   #1a1a2e;
            --bcgx-grey:   #F5F5F5;
            --bcgx-slate:  #4A5568;
        }
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213E 50%, #0F3460 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
        }
        .main-header h1 { color: #00A651; margin: 0 0 0.25rem 0; font-size: 2.2rem; }
        .main-header h3 { color: #CBD5E0; margin: 0; font-weight: 400; font-size: 1.1rem; }
        .main-header .subtitle { color: #A0AEC0; font-size: 0.9rem; margin-top: 0.5rem; }
        .section-card {
            background: #FAFAFA;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .metric-note {
            background: #EBF8F1;
            border-left: 4px solid #00A651;
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
            color: #276749;
            margin-top: 1.5rem;
        }
        div[data-testid="metric-container"] {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>NovaMart Analytics &nbsp;&mdash;&nbsp; BCG X</h1>
        <h3>Retail Intelligence Platform &mdash; Fortune 500 Consulting Engagement</h3>
        <div class="subtitle">
            Powered by BCG X &bull; Confidential &bull; For NovaMart Leadership Use Only
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
_data_loaded = False
_tx = None
_stores = None
_customers = None

try:
    import warnings

    warnings.filterwarnings("ignore")
    from bcgx.data.loader import DataLoader

    _loader = DataLoader()
    _data = _loader.load_all()
    _tx = _data["transactions"]
    _stores = _data["stores"]
    _customers = _data["customers"]
    _data_loaded = True
except Exception:
    pass

# ── KPI Metric Cards ──────────────────────────────────────────────────────────
st.subheader("Key Performance Indicators")

m1, m2, m3, m4 = st.columns(4)

if _data_loaded and _tx is not None:
    total_rev = _tx["gross_revenue"].sum()
    total_profit = _tx["gross_profit"].sum()
    margin = total_profit / total_rev * 100 if total_rev > 0 else 0
    n_months = _tx["date"].dt.to_period("M").nunique()
    monthly_rev = total_rev / n_months

    with m1:
        st.metric("Total Revenue (36m)", f"${total_rev/1e6:.1f}M", help="Gross revenue over 36 months")
    with m2:
        st.metric("Avg Gross Margin", f"{margin:.1f}%", help="Average gross margin across all transactions")
    with m3:
        st.metric("Total Stores", f"{len(_stores):,}", help="Active NovaMart store locations")
    with m4:
        st.metric("Total Customers", f"{len(_customers):,}", help="Unique customers in the loyalty programme")

    st.markdown(
        """<div class="metric-note">
        <strong>Data loaded successfully.</strong>
        Use the sidebar to navigate to individual analysis modules.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    with m1:
        st.metric("Total Revenue (36m)", "—")
    with m2:
        st.metric("Avg Gross Margin", "—")
    with m3:
        st.metric("Total Stores", "—")
    with m4:
        st.metric("Total Customers", "—")

    st.warning(
        "Data not yet generated. Run `python scripts/generate_data.py` to populate this dashboard."
    )

# ── Navigation ─────────────────────────────────────────────────────────────────
st.divider()
col_desc, col_nav = st.columns([2, 1])

with col_desc:
    st.markdown(
        """<div class="section-card">
        <h4 style="margin-top:0; color:#1a1a2e;">About This Platform</h4>
        <p>The <strong>BCG X Analytics Accelerator</strong> surfaces insights across four pillars:</p>
        <ul>
            <li><strong>Customer Intelligence</strong> — churn prediction, CLV segmentation</li>
            <li><strong>Store Performance</strong> — ranking, anomaly detection, benchmarking</li>
            <li><strong>Pricing &amp; Promotion</strong> — elasticity modelling, margin recovery</li>
            <li><strong>Marketing Mix</strong> — spend attribution, ROI optimisation, media planning</li>
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )

with col_nav:
    st.markdown(
        """<div class="section-card">
        <h4 style="margin-top:0; color:#1a1a2e;">Dashboard Pages</h4>
        <ul style="color:#4A5568; font-size:0.9rem;">
            <li>01 Executive Overview</li>
            <li>02 Data Audit</li>
            <li>03 EDA</li>
            <li>04 Statistical Analysis</li>
            <li>05 Model Performance</li>
            <li>06 Scenario Simulator</li>
            <li>07 Recommendations</li>
            <li>08 AI Copilot</li>
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "BCG X Analytics Accelerator v1.0.0  |  "
    "Confidential — NovaMart Internal Use Only  |  "
    "2024 Boston Consulting Group"
)
