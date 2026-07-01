"""Scenario Simulator — Interactive what-if analysis for NovaMart leadership."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Scenario Simulator | NovaMart", layout="wide")

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from dashboard.components.styles import inject_css, section_header, insight, kpi_card, page_header
inject_css()

st.title("Scenario Simulator")
st.caption("Analytics Simulation · BCG X–Inspired Methodology")

# ── Sidebar: Scenario selector ────────────────────────────────────────────────
scenario = st.sidebar.radio(
    "Scenario Type",
    ["Price Change", "Marketing Reallocation", "Churn Reduction", "Store Investment"],
    index=0,
)

# ── Load engine lazily ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading simulation engine...")
def _get_engine():
    from bcgx.simulation.engine import SimulationEngine

    return SimulationEngine()


try:
    engine = _get_engine()
    ENGINE_OK = True
except Exception as e:
    st.error(f"Simulation engine unavailable: {e}\nRun `python scripts/generate_data.py` first.")
    ENGINE_OK = False

if not ENGINE_OK:
    st.stop()


def _delta_colour(val: float) -> str:
    return "normal" if val >= 0 else "inverse"


def _confidence_gauge(confidence: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={"text": "Model Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00A651"},
                "steps": [
                    {"range": [0, 50], "color": "#FED7D7"},
                    {"range": [50, 75], "color": "#FEFCBF"},
                    {"range": [75, 100], "color": "#C6F6D5"},
                ],
                "threshold": {
                    "line": {"color": "#1a1a2e", "width": 3},
                    "thickness": 0.75,
                    "value": confidence * 100,
                },
            },
            number={"suffix": "%", "font": {"size": 28}},
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="white")
    return fig


def _show_results(result, implementation_cost: float = 0.0) -> None:
    """Render simulation results in the right panel."""
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Projected Revenue",
        f"${result.projected_revenue/1e6:.2f}M",
        delta=f"{result.revenue_delta_pct:+.2f}%",
        delta_color=_delta_colour(result.revenue_delta),
        help="Annual projected revenue",
    )
    col2.metric(
        "Projected Profit",
        f"${result.projected_profit/1e6:.2f}M",
        delta=f"{result.profit_delta_pct:+.2f}%",
        delta_color=_delta_colour(result.profit_delta),
        help="Annual projected gross profit",
    )
    col3.metric(
        "Revenue Delta",
        f"${result.revenue_delta/1e6:+.2f}M",
        help="Annual revenue change vs baseline",
    )

    col4, col5, col6 = st.columns(3)
    col4.metric("Baseline Revenue", f"${result.baseline_revenue/1e6:.2f}M")
    col5.metric("Baseline Profit", f"${result.baseline_profit/1e6:.2f}M")
    col6.metric("Baseline Margin", f"{result.baseline_margin_pct:.2f}%")

    st.divider()

    gauge_col, info_col = st.columns([1, 2])
    with gauge_col:
        st.plotly_chart(_confidence_gauge(result.confidence_level), use_container_width=True)
    with info_col:
        st.markdown(f"**Projected Margin:** {result.projected_margin_pct:.2f}%")
        st.markdown(f"**Profit Delta:** ${result.profit_delta/1e6:+.2f}M")
        st.markdown(f"**Timeline:** {result.timeline_months} months")
        if implementation_cost > 0 and result.profit_delta > 0:
            roi = result.profit_delta / implementation_cost
            st.markdown(f"**ROI:** {roi:.1f}x")

    with st.expander("Assumptions & Risks", expanded=False):
        st.markdown("**Assumptions:**")
        for a in result.assumptions:
            st.markdown(f"- {a}")
        st.markdown("**Risks:**")
        for r in result.risks:
            st.markdown(f"- {r}")


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE CHANGE SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════
if scenario == "Price Change":
    st.subheader("Price Change Simulation")
    st.markdown(
        "Simulate the revenue and profit impact of changing prices in a specific product category, "
        "accounting for consumer price elasticity."
    )

    col_ctl, col_res = st.columns([1, 2])
    with col_ctl:
        category = st.selectbox(
            "Product Category",
            ["Electronics", "Food & Beverage", "Health & Beauty", "Apparel", "Home & Garden", "Sports", "Toys"],
        )
        price_change = st.slider("Price Change (%)", -20, 20, 5, step=1)
        price_change_frac = price_change / 100

        # Show elasticity context
        elasticity_map = {
            "Electronics": -1.95, "Food & Beverage": -0.45, "Health & Beauty": -0.80,
            "Apparel": -1.25, "Home & Garden": -1.10, "Sports": -1.50, "Toys": -1.65,
        }
        elasticity = elasticity_map.get(category, -1.0)
        inelastic = abs(elasticity) < 1.0
        label = "INELASTIC (price increase grows revenue)" if inelastic else "ELASTIC (price increase reduces revenue)"
        colour = "success" if (inelastic and price_change > 0) or (not inelastic and price_change < 0) else "warning"
        getattr(st, colour)(f"Elasticity: {elasticity:.2f} — {label}")

        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    with col_res:
        if run_btn:
            with st.spinner("Running simulation..."):
                try:
                    result = engine.simulate_price_change(category, price_change_frac)
                    _show_results(result)
                except Exception as e:
                    st.error(f"Simulation error: {e}")
        else:
            st.info("Configure the scenario parameters and click **Run Simulation**.")

# ═══════════════════════════════════════════════════════════════════════════════
# MARKETING REALLOCATION SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════
elif scenario == "Marketing Reallocation":
    st.subheader("Marketing Reallocation Simulation")
    st.markdown(
        "Simulate the revenue impact of reallocating the marketing budget across channels. "
        "Adjust sliders to set the new % allocation for each channel."
    )

    col_ctl, col_res = st.columns([1, 2])
    with col_ctl:
        store_format = st.selectbox("Store Format", ["all", "urban", "rural", "suburban"])

        st.markdown("**Channel Allocation (must sum to 100%)**")
        digital = st.slider("Digital (%)", 0, 100, 35)
        tv = st.slider("TV (%)", 0, 100, 25)
        email = st.slider("Email (%)", 0, 100, 20)
        print_ = st.slider("Print (%)", 0, 100, 12)
        instore = 100 - digital - tv - email - print_

        total = digital + tv + email + print_
        if total > 100:
            st.error(f"Allocation exceeds 100% ({total}%). Adjust sliders.")
            instore = 0
        elif instore < 0:
            st.error("Remaining instore_promo would be negative. Reduce other channels.")
            instore = 0
        else:
            st.metric("In-Store Promo (auto)", f"{instore}%")

        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    with col_res:
        if run_btn and total <= 100:
            reallocation = {
                "digital": digital / 100,
                "tv": tv / 100,
                "email": email / 100,
                "print": print_ / 100,
                "instore_promo": instore / 100,
            }
            with st.spinner("Running simulation..."):
                try:
                    result = engine.simulate_marketing_reallocation(reallocation, store_format=store_format)
                    _show_results(result)
                except Exception as e:
                    st.error(f"Simulation error: {e}")
        elif run_btn:
            st.error("Fix allocation (must be <= 100%) before running.")
        else:
            st.info("Configure the scenario parameters and click **Run Simulation**.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHURN REDUCTION SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════
elif scenario == "Churn Reduction":
    st.subheader("Churn Reduction Simulation")
    st.markdown(
        "Simulate the revenue and profit impact of a loyalty programme designed to reduce "
        "customer churn in a specific loyalty tier."
    )

    col_ctl, col_res = st.columns([1, 2])
    with col_ctl:
        segment = st.selectbox("Target Loyalty Tier", ["Silver", "Bronze", "Gold", "All"])
        reduction_pct = st.slider("Churn Reduction (%)", 0, 100, 30, step=5)
        intervention_cost = st.number_input(
            "Intervention Cost ($)", min_value=0, max_value=10_000_000, value=500_000, step=50_000
        )
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    with col_res:
        if run_btn:
            with st.spinner("Running simulation..."):
                try:
                    result = engine.simulate_churn_reduction(
                        segment, reduction_pct / 100, float(intervention_cost)
                    )
                    _show_results(result, implementation_cost=float(intervention_cost))
                except Exception as e:
                    st.error(f"Simulation error: {e}")
        else:
            st.info("Configure the scenario parameters and click **Run Simulation**.")

# ═══════════════════════════════════════════════════════════════════════════════
# STORE INVESTMENT SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════
elif scenario == "Store Investment":
    st.subheader("Store Investment Simulation")
    st.markdown(
        "Simulate the revenue uplift from investing capital in the worst-performing stores "
        "to bring them up to median performance levels."
    )

    col_ctl, col_res = st.columns([1, 2])
    with col_ctl:
        n_stores = st.slider("Number of Stores to Invest In", 5, 200, 50, step=5)
        investment = st.number_input(
            "Investment per Store ($)", min_value=10_000, max_value=1_000_000, value=150_000, step=10_000
        )
        improvement_pct = st.slider("Expected Performance Improvement (%)", 1, 50, 15, step=1)
        total_inv = n_stores * investment
        st.metric("Total Investment", f"${total_inv/1e6:.1f}M")
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    with col_res:
        if run_btn:
            with st.spinner("Running simulation..."):
                try:
                    result = engine.simulate_store_investment(
                        n_stores, float(investment), improvement_pct / 100
                    )
                    _show_results(result, implementation_cost=float(total_inv))
                except Exception as e:
                    st.error(f"Simulation error: {e}")
        else:
            st.info("Configure the scenario parameters and click **Run Simulation**.")
