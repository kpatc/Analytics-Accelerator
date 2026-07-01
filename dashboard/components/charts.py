"""Reusable Plotly chart functions for the NovaMart dashboard.

All charts use Plotly (graph_objects or express) — NOT matplotlib.
Each function returns a go.Figure ready for st.plotly_chart().
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# BCG X brand palette
BCGX_GREEN = "#00A651"
BCGX_DARK = "#1a1a2e"
BCGX_GREY = "#F5F5F5"
BCGX_SLATE = "#4A5568"
BCGX_BLUE = "#0F3460"

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, Arial, sans-serif", color=BCGX_SLATE),
    margin=dict(l=40, r=20, t=60, b=40),
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
    fig.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["revenue"] / 1e6,
            name="Revenue ($M)",
            line=dict(color=BCGX_BLUE, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(15,52,96,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["profit"] / 1e6,
            name="Gross Profit ($M)",
            line=dict(color=BCGX_GREEN, width=2.5),
        )
    )
    fig.update_layout(
        title="Is NovaMart's revenue and profit trending in the right direction?",
        xaxis_title="Month",
        yaxis_title="USD (millions)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_LAYOUT_DEFAULTS,
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
    fig.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["margin_pct"],
            name="Gross Margin %",
            line=dict(color=BCGX_GREEN, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,166,81,0.10)",
        )
    )
    # Add reference line at overall average
    avg = monthly["margin_pct"].mean()
    fig.add_hline(
        y=avg,
        line_dash="dash",
        line_color=BCGX_SLATE,
        annotation_text=f"36-month avg: {avg:.1f}%",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Is margin declining? (Gross Margin % over 36 months)",
        xaxis_title="Month",
        yaxis_title="Gross Margin %",
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

    colour_map = {"A": BCGX_GREEN, "B": "#F6AD55", "C": "#FC4E03"}

    fig = px.scatter(
        merged,
        x="sq_footage",
        y="margin_pct",
        color="performance_cluster",
        color_discrete_map=colour_map,
        hover_data=["store_id", "city", "store_format", "manager_tenure_years"],
        labels={
            "sq_footage": "Store Size (sq ft)",
            "margin_pct": "Gross Margin %",
            "performance_cluster": "Cluster",
        },
    )
    fig.update_layout(
        title="Which stores are over- and under-performing? (Cluster A=best, C=worst)",
        **_LAYOUT_DEFAULTS,
    )
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

    colour_seq = [BCGX_GREEN, BCGX_BLUE, "#F6AD55", "#FC4E03", BCGX_SLATE]

    fig = go.Figure(
        go.Pie(
            labels=counts["segment"],
            values=counts["count"],
            hole=0.55,
            marker=dict(colors=colour_seq[: len(counts)]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="How are customers distributed across value segments?",
        **_LAYOUT_DEFAULTS,
    )
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

    fig = go.Figure(
        go.Bar(
            x=top15.values,
            y=top15.index,
            orientation="h",
            marker=dict(
                color=top15.values,
                colorscale=[[0, "#CBD5E0"], [1, BCGX_GREEN]],
                showscale=False,
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Feature Importance",
        yaxis_title="",
        **_LAYOUT_DEFAULTS,
    )
    return fig
