"""Reusable KPI card component for the NovaMart dashboard."""

from __future__ import annotations

import streamlit as st


def render_kpi_card(
    title: str,
    value: str,
    delta: str | None = None,
    delta_positive: bool = True,
    help_text: str | None = None,
) -> None:
    """Render a styled KPI metric card using st.metric.

    Args:
        title: Metric label displayed above the value.
        value: Primary metric value as a pre-formatted string.
        delta: Change indicator string (e.g. "+5.2%"). None to hide.
        delta_positive: Whether a positive delta is good (green) or bad (red).
        help_text: Optional tooltip text.
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color="normal" if delta_positive else "inverse",
        help=help_text,
    )
