"""Executive Analytics Copilot — Claude-powered business Q&A interface.

Replaces the keyword-matching stub with a real agentic copilot that calls
analytics tools to retrieve grounded NovaMart data before answering.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(page_title="AI Copilot | NovaMart", layout="wide")

_repo = Path(__file__).resolve().parents[2]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from dashboard.components.styles import inject_css, section_header, insight, kpi_card, page_header
inject_css()

# ── Page header ────────────────────────────────────────────────────────────────
st.title("Executive Analytics Copilot")
st.caption("Analytics Simulation · BCG X–Inspired Methodology · Powered by Anthropic Claude")

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a1a2e, #0F3460); padding: 1.5rem;
    border-radius: 12px; color: white; margin-bottom: 1rem;">
        <h3 style="color: #00A651; margin: 0 0 0.5rem 0;">
            Ask any business question about NovaMart
        </h3>
        <p style="color: #CBD5E0; margin: 0;">
            The AI Copilot synthesises insights from 36 months of transaction data,
            ML model outputs, and advanced analytics frameworks inspired by real consulting delivery. Every answer is grounded
            in actual NovaMart data — retrieved in real time via analytics tools.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Copilot initialisation ─────────────────────────────────────────────────────
from bcgx.copilot.agent import ExecutiveCopilot
from bcgx.copilot.prompts import EXAMPLE_QUESTIONS

copilot = ExecutiveCopilot()
configured = copilot.is_configured()

# Configuration status banner
if configured:
    st.success("Anthropic API key detected — Full AI Copilot is active.")
else:
    st.warning(
        "Running without an API key. "
        "Set `ANTHROPIC_API_KEY` in your `.env` file to enable the AI Copilot. "
        "Questions will return setup instructions until the key is configured."
    )

# ── Example questions ──────────────────────────────────────────────────────────
st.subheader("Example Questions")
st.caption("Click any question to ask it immediately.")

cols = st.columns(3)
for i, (col, q) in enumerate(zip(cols * 3, EXAMPLE_QUESTIONS)):
    with col:
        if st.button(q, use_container_width=True, key=f"example_{i}"):
            st.session_state["copilot_prefill"] = q

st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_called"):
            st.caption(f"Data retrieved from: {', '.join(msg['tools_called'])}")

# ── Chat input ─────────────────────────────────────────────────────────────────
# Pre-populate from example button click
prefill = st.session_state.pop("copilot_prefill", "")

question = st.chat_input(
    "Ask a business question about NovaMart performance...",
    key="copilot_chat_input",
)

# Treat example button click as a question
if not question and prefill:
    question = prefill

# ── Process question ───────────────────────────────────────────────────────────
if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get copilot response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving analytics and generating answer..."):
            # Build conversation history from prior messages (text only)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # exclude the question just added
                if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
            ]

            response = copilot.ask(question, history)
            response_text = response.answer

        st.markdown(response_text)

        if response.tools_called:
            st.caption(f"Data retrieved from: {', '.join(response.tools_called)}")

    # Store assistant message with metadata
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "tools_called": response.tools_called,
        }
    )

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Copilot Controls")

    if st.session_state.messages:
        if st.button("Clear Conversation", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.subheader("Available Analytics")
    st.markdown(
        """
        The copilot has access to:
        - **Financial KPIs** — revenue, margin, profit trends
        - **Churn Analysis** — by loyalty tier, month-24 inflection
        - **Store Performance** — cluster A/B/C breakdown
        - **Marketing ROI** — channel × format matrix
        - **Pricing Analysis** — elasticity by category, private label
        - **Recommendations** — RICE-scored strategic roadmap
        - **Scenario Simulator** — price, marketing, churn, store what-ifs
        """
    )

    st.divider()
    st.caption(
        "Analytics Simulation | Powered by Anthropic Claude | "
        "Data: NovaMart synthetic 36-month retail dataset"
    )
