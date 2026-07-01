"""Global design system for the NovaMart dashboard.

Single source of truth for colours, typography, and CSS.
Call inject_global_css() at the top of every page.
"""

from __future__ import annotations

import streamlit as st

# ── Brand tokens ──────────────────────────────────────────────────────────────
GREEN = "#00A651"
DARK = "#0D1117"
NAVY = "#161B27"
CARD = "#1C2333"
BORDER = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
WHITE = "#FFFFFF"
RED = "#FF6B6B"
AMBER = "#FFA500"

_CSS = """
<style>
/* ── Reset & base ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App background */
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B27 !important;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] * {
    color: #E6EDF3 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stRadio > div {
    background: transparent;
}

/* Main content area */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

/* ── Typography ───────────────────────────────────────────────────── */
h1 { color: #E6EDF3 !important; font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: -0.02em; margin-bottom: 0.25rem !important; }
h2 { color: #E6EDF3 !important; font-size: 1.2rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }
h3 { color: #C9D1D9 !important; font-size: 1rem !important; font-weight: 500 !important; }
p, li { color: #C9D1D9; line-height: 1.6; }

/* ── KPI Cards ────────────────────────────────────────────────────── */
.kpi-card {
    background: #1C2333;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00A651, #00C762);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.kpi-delta-pos {
    font-size: 0.78rem;
    font-weight: 500;
    color: #00A651;
    margin-top: 0.35rem;
}
.kpi-delta-neg {
    font-size: 0.78rem;
    font-weight: 500;
    color: #FF6B6B;
    margin-top: 0.35rem;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #8B949E;
    margin-top: 0.25rem;
}

/* ── Section header ───────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.75rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #21262D;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00A651;
    flex-shrink: 0;
}
.section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #C9D1D9;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Insight box ──────────────────────────────────────────────────── */
.insight-box {
    background: rgba(0, 166, 81, 0.08);
    border: 1px solid rgba(0, 166, 81, 0.3);
    border-left: 3px solid #00A651;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #C9D1D9;
    line-height: 1.6;
}
.insight-box strong { color: #00A651; }

/* ── Warning box ──────────────────────────────────────────────────── */
.warn-box {
    background: rgba(255, 107, 107, 0.08);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-left: 3px solid #FF6B6B;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #C9D1D9;
}
.warn-box strong { color: #FF6B6B; }

/* ── Badge pills ──────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-green  { background: rgba(0,166,81,0.15);  color: #00A651; border: 1px solid rgba(0,166,81,0.3); }
.badge-red    { background: rgba(255,107,107,0.15); color: #FF6B6B; border: 1px solid rgba(255,107,107,0.3); }
.badge-amber  { background: rgba(255,165,0,0.15);  color: #FFA500; border: 1px solid rgba(255,165,0,0.3); }
.badge-grey   { background: rgba(139,148,158,0.15); color: #8B949E; border: 1px solid rgba(139,148,158,0.3); }
.badge-blue   { background: rgba(88,166,255,0.15); color: #58A6FF; border: 1px solid rgba(88,166,255,0.3); }

/* ── Table styling ────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── Streamlit widget overrides ───────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #1C2333;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
div[data-testid="stMetric"] label {
    color: #8B949E !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* Selectbox & radio */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #1C2333 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    color: #E6EDF3 !important;
}
.stRadio > div {
    gap: 0.4rem;
}
.stRadio label {
    color: #C9D1D9 !important;
}

/* Slider */
.stSlider > div > div > div {
    background: #30363D !important;
}

/* Buttons */
.stButton > button {
    background: #00A651 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Expander */
details {
    background: #1C2333 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
}
summary { color: #C9D1D9 !important; font-weight: 500 !important; }

/* Divider */
hr { border-color: #21262D !important; margin: 1.25rem 0 !important; }

/* Info / success / error / warning overrides */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-width: 1px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #00A651 !important; }

/* Page nav sidebar links */
.css-1544g2n { padding: 1rem 0.75rem; }

/* Tab styling */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #8B949E !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00A651 !important;
    border-bottom: 2px solid #00A651 !important;
}

/* Chat input */
.stChatInput textarea {
    background: #1C2333 !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 10px !important;
}
.stChatMessage {
    background: #1C2333 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
}

/* Plotly chart background fix */
.js-plotly-plot .plotly .modebar {
    background: transparent !important;
}

/* Number input */
input[type="number"] {
    background: #1C2333 !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 6px !important;
}
</style>
"""


def inject_css() -> None:
    """Inject the global design system CSS. Call once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str | None = None, positive: bool = True, sub: str | None = None) -> None:
    """Render a styled dark KPI card."""
    delta_html = ""
    if delta:
        cls = "kpi-delta-pos" if positive else "kpi-delta-neg"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}{sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    """Render a subtle section divider with label."""
    st.markdown(
        f"""<div class="section-header">
            <div class="section-dot"></div>
            <div class="section-title">{title}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    """Render a green insight callout box."""
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


def warning_box(text: str) -> None:
    """Render a red warning callout box."""
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)


def badge(text: str, color: str = "green") -> str:
    """Return HTML for a badge pill. color: green | red | amber | grey | blue"""
    return f'<span class="badge badge-{color}">{text}</span>'


def page_header(title: str, subtitle: str = "") -> None:
    """Render the top page header with optional subtitle."""
    sub_html = f'<p style="color:#8B949E;font-size:0.82rem;margin:0.25rem 0 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""<div style="margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #21262D;">
            <h1 style="margin:0;">{title}</h1>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )
