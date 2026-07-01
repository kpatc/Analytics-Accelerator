"""Strategic Recommendations — NovaMart prioritised action agenda."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recommendations | NovaMart", layout="wide")

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

st.title("Strategic Recommendations")
st.caption("BCG X Analytics Accelerator | Confidential — Evidence-Backed Action Agenda")

# ── Load recommendations ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating recommendations...")
def _load_recommendations():
    from bcgx.recommendations.engine import RecommendationEngine
    from bcgx.recommendations.prioritizer import RecommendationPrioritizer

    engine = RecommendationEngine()
    recs = engine.generate_all()
    prioritizer = RecommendationPrioritizer()
    ranked = prioritizer.prioritize(recs)
    df = prioritizer.to_dataframe(ranked)
    return ranked, df


try:
    ranked_recs, recs_df = _load_recommendations()
    RECS_OK = True
except Exception as e:
    st.error(f"Could not generate recommendations: {e}\nRun `python scripts/generate_data.py` first.")
    RECS_OK = False

if not RECS_OK:
    st.stop()

# ── Summary KPIs ───────────────────────────────────────────────────────────────
total_rev = sum(r.expected_revenue_impact_usd for r in ranked_recs)
total_profit = sum(r.expected_profit_impact_usd for r in ranked_recs)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Recommendations", len(ranked_recs))
k2.metric("Total Revenue Opportunity", f"${total_rev/1e6:.1f}M/yr")
k3.metric("Total Profit Opportunity", f"${total_profit/1e6:.1f}M/yr")
k4.metric(
    "Critical Priority Items",
    sum(1 for r in ranked_recs if r.priority.value == "Critical"),
)

st.divider()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filter Recommendations")
priority_filter = st.sidebar.multiselect(
    "Priority",
    ["Critical", "High", "Medium", "Low"],
    default=["Critical", "High", "Medium", "Low"],
)
category_filter = st.sidebar.multiselect(
    "Category",
    sorted(set(r.category for r in ranked_recs)),
    default=sorted(set(r.category for r in ranked_recs)),
)
timeline_filter = st.sidebar.multiselect(
    "Timeline",
    ["0-30 days", "1-3 months", "3-6 months", "6-12 months"],
    default=["0-30 days", "1-3 months", "3-6 months", "6-12 months"],
)

filtered = [
    r for r in ranked_recs
    if r.priority.value in priority_filter
    and r.category in category_filter
    and r.timeline.value in timeline_filter
]

st.subheader(f"Recommendations ({len(filtered)} shown)")

# ── Summary table ──────────────────────────────────────────────────────────────
PRIORITY_EMOJI = {"Critical": "🔴", "High": "🟡", "Medium": "⚪", "Low": "⚫"}

table_rows = [
    {
        "Rank": i + 1,
        "ID": r.id,
        "Title": r.title,
        "Category": r.category,
        "Priority": f"{PRIORITY_EMOJI.get(r.priority.value, '')} {r.priority.value}",
        "Revenue Impact": f"${r.expected_revenue_impact_usd/1e6:.1f}M",
        "Profit Impact": f"${r.expected_profit_impact_usd/1e6:.1f}M",
        "ROI": f"{r.roi:.1f}x",
        "RICE Score": f"{r.rice_score:.1f}",
        "Timeline": r.timeline.value,
        "Difficulty": r.difficulty.value,
    }
    for i, r in enumerate(filtered)
]

if table_rows:
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
else:
    st.warning("No recommendations match the selected filters.")

st.divider()

# ── Expandable detail cards ────────────────────────────────────────────────────
st.subheader("Recommendation Details")

for r in filtered:
    priority_colour = {
        "Critical": "#FC4E03", "High": "#F6AD55", "Medium": "#4A5568", "Low": "#CBD5E0"
    }.get(r.priority.value, "#4A5568")

    with st.expander(f"**{r.id}** — {r.title}  [{r.priority.value}]", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue Impact", f"${r.expected_revenue_impact_usd/1e6:.1f}M/yr")
        col2.metric("Profit Impact", f"${r.expected_profit_impact_usd/1e6:.1f}M/yr")
        col3.metric("ROI", f"{r.roi:.1f}x")

        col4, col5, col6 = st.columns(3)
        col4.metric("RICE Score", f"{r.rice_score:.1f}")
        col5.metric("Confidence", f"{r.confidence:.0%}")
        col6.metric("Effort", f"{r.implementation_effort_weeks} weeks")

        st.markdown(f"**Description:** {r.description}")
        st.markdown(f"**Evidence:** {r.evidence}")
        st.markdown(f"**Risk:** {r.risk}")

        if r.kpis_to_track:
            st.markdown("**KPIs to Track:**")
            for kpi in r.kpis_to_track:
                st.markdown(f"- {kpi}")

        if r.dependencies:
            st.markdown(f"**Dependencies:** {', '.join(r.dependencies)}")
