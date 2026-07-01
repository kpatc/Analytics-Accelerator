"""Reusable Plotly chart functions for the NovaMart dashboard.

All charts use Plotly (graph_objects or express) — NOT matplotlib.
Each function returns a go.Figure ready for st.plotly_chart().
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dark design system palette
BCGX_GREEN = "#00A651"
BCGX_BLUE = "#58A6FF"
BCGX_AMBER = "#FFA500"
BCGX_RED = "#FF6B6B"
BCGX_PURPLE = "#BC8CFF"
BG = "#0D1117"
CARD = "#1C2333"
BORDER = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color=MUTED),
    margin=dict(l=48, r=24, t=52, b=40),
    title_font=dict(size=13, color=TEXT, family="Inter, sans-serif"),
    xaxis=dict(
        gridcolor="#21262D",
        linecolor=BORDER,
        tickcolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    ),
    yaxis=dict(
        gridcolor="#21262D",
        linecolor=BORDER,
        tickcolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    hoverlabel=dict(
        bgcolor="#161B27",
        bordercolor=BORDER,
        font=dict(color=TEXT, size=12),
    ),
)


def revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly Revenue and Profit dual-axis trend chart.

    Args:
        df: DataFrame with columns [year_month, gross_revenue, gross_profit].

    Returns:
        Plotly figure.
    """
    monthly = (
        df.groupby("year_month")
        .agg(revenue=("gross_revenue", "sum"), profit=("gross_profit", "sum"))
        .reset_index()
        .sort_values("year_month")
    )
    monthly["margin_pct"] = monthly["profit"] / monthly["revenue"] * 100

    fig = go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["revenue"] / 1e6,
        name="Revenue ($M)", line=dict(color=BCGX_BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["profit"] / 1e6,
        name="Gross Profit ($M)", line=dict(color=BCGX_GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(0,166,81,0.07)",
    ))
    fig.update_layout(
        title="Revenue & Profit Trend",
        xaxis_title=None, yaxis_title="USD (millions)",
        hovermode="x unified", **_LAYOUT_DEFAULTS,
    )
    return fig


def margin_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Gross margin % trend over time.

    Args:
        df: DataFrame with columns [year_month, gross_revenue, gross_profit].

    Returns:
        Plotly figure.
    """
    monthly = (
        df.groupby("year_month")
        .agg(revenue=("gross_revenue", "sum"), profit=("gross_profit", "sum"))
        .reset_index()
        .sort_values("year_month")
    )
    monthly["margin_pct"] = monthly["profit"] / monthly["revenue"] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["margin_pct"],
        name="Gross Margin %", line=dict(color=BCGX_GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(0,166,81,0.08)",
    ))
    avg = monthly["margin_pct"].mean()
    fig.add_hline(
        y=avg, line_dash="dot", line_color=MUTED, line_width=1,
        annotation_text=f"avg {avg:.1f}%",
        annotation_font=dict(color=MUTED, size=11),
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Gross Margin % — 36-Month Trend",
        xaxis_title=None, yaxis_title="Gross Margin %",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def store_scatter_chart(stores: pd.DataFrame, transactions: pd.DataFrame) -> go.Figure:
    """Store performance scatter: sq_footage vs profit margin, coloured by cluster.

    Args:
        stores: DataFrame with store metadata including performance_cluster.
        transactions: DataFrame with gross_revenue and gross_profit.

    Returns:
        Plotly figure.
    """
    store_tx = (
        transactions.groupby("store_id")
        .agg(revenue=("gross_revenue", "sum"), profit=("gross_profit", "sum"))
        .reset_index()
    )
    store_tx["margin_pct"] = store_tx["profit"] / store_tx["revenue"] * 100
    merged = stores.merge(store_tx, on="store_id", how="left")

    colour_map = {"A": BCGX_GREEN, "B": BCGX_AMBER, "C": BCGX_RED}

    fig = px.scatter(
        merged, x="sq_footage", y="margin_pct",
        color="performance_cluster", color_discrete_map=colour_map,
        hover_data=["store_id", "store_format", "manager_tenure_years"],
        labels={"sq_footage": "Store Size (sq ft)", "margin_pct": "Gross Margin %", "performance_cluster": "Cluster"},
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_layout(title="Store Portfolio — Size vs Margin by Cluster", **_LAYOUT_DEFAULTS)
    return fig


def rfm_donut_chart(rfm: pd.DataFrame) -> go.Figure:
    """Customer RFM segment distribution donut chart.

    Args:
        rfm: DataFrame with at least 'loyalty_tier' or 'segment' column and counts.

    Returns:
        Plotly figure.
    """
    if "loyalty_tier" in rfm.columns:
        group_col = "loyalty_tier"
    elif "segment" in rfm.columns:
        group_col = "segment"
    else:
        group_col = rfm.columns[0]

    counts = rfm[group_col].value_counts().reset_index()
    counts.columns = ["segment", "count"]

    colour_seq = [BCGX_GREEN, BCGX_BLUE, BCGX_AMBER, BCGX_RED, BCGX_PURPLE]

    fig = go.Figure(go.Pie(
        labels=counts["segment"], values=counts["count"], hole=0.6,
        marker=dict(colors=colour_seq[: len(counts)], line=dict(color=CARD, width=2)),
        textinfo="label+percent", textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(title="Customer Segment Distribution", **_LAYOUT_DEFAULTS)
    return fig


def feature_importance_chart(feature_importance: pd.Series, title: str) -> go.Figure:
    """Horizontal bar chart of feature importances.

    Args:
        feature_importance: Series with feature names as index and importance scores as values.
        title: Chart title (the business question being answered).

    Returns:
        Plotly figure.
    """
    top15 = feature_importance.nlargest(15).sort_values()

    fig = go.Figure(go.Bar(
        x=top15.values, y=top15.index, orientation="h",
        marker=dict(
            color=top15.values,
            colorscale=[[0, "#1C2333"], [1, BCGX_GREEN]],
            showscale=False,
        ),
    ))
    fig.update_layout(title=title, xaxis_title="Importance", yaxis_title=None, **_LAYOUT_DEFAULTS)
    return fig
