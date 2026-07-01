"""Run exploratory data analysis for the NovaMart dataset.

Milestone 2: Data Audit & EDA
Orchestrates the full EDA pipeline using InsightExtractor and outputs:
- A rich console summary
- JSON insights to data/outputs/eda_insights.json

Usage:
    python scripts/run_eda.py [OPTIONS]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src/ is on the path when run as a script (not installed as a package)
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bcgx.data.loader import DataLoader
from bcgx.eda.insights import InsightExtractor

console = Console()
app = typer.Typer(help="Run exploratory data analysis on NovaMart datasets.")


@app.command()
def main(
    data_dir: str = typer.Option("data/raw", help="Directory containing raw parquet files"),
    output_path: str = typer.Option(
        "data/outputs/eda_insights.json", help="JSON output path for EDA insights"
    ),
) -> None:
    """Run the full NovaMart EDA pipeline.

    Loads all raw datasets, runs univariate, bivariate, and temporal analyses,
    saves insights to JSON, and prints an executive summary to the console.
    """
    t0 = time.perf_counter()

    console.print(
        Panel.fit(
            "[bold cyan]NovaMart EDA Pipeline[/bold cyan]\n"
            "[dim]BCG X Analytics Accelerator — Milestone 2[/dim]",
            border_style="cyan",
        )
    )

    # --- Load data ---
    console.print("\n[bold]Step 1: Loading datasets...[/bold]")
    try:
        loader = DataLoader(data_dir)
        data = loader.load_all()
    except FileNotFoundError as exc:
        console.print(f"[bold red]ERROR:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    for name, df in data.items():
        console.print(f"  [green]✓[/green] {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # --- Run EDA ---
    console.print("\n[bold]Step 2: Running EDA analyses...[/bold]")
    extractor = InsightExtractor()
    summary = extractor.run(data)

    # --- Save JSON ---
    console.print("\n[bold]Step 3: Saving insights to JSON...[/bold]")
    extractor.save_json(summary, output_path)
    console.print(f"  [green]✓[/green] Saved to [italic]{output_path}[/italic]")

    # --- Print executive summary ---
    elapsed = time.perf_counter() - t0
    _print_summary(summary)
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")


def _print_summary(summary: object) -> None:
    """Print a rich console summary of the EDA results."""
    from bcgx.eda.insights import EDAInsightSummary

    if not isinstance(summary, EDAInsightSummary):
        return

    # --- Headline financials ---
    console.print(
        Panel.fit(
            f"[bold]Portfolio Financials[/bold]\n"
            f"  Total Revenue:   [bold green]${summary.total_revenue / 1e6:,.1f}M[/bold green]\n"
            f"  Total Profit:    [bold green]${summary.total_profit / 1e6:,.1f}M[/bold green]\n"
            f"  Avg Gross Margin: [bold]{ summary.avg_gross_margin_pct:.1f}%[/bold]\n"
            f"  Margin Trend:    [{'bold red' if summary.margin_trend_pct_change < 0 else 'bold green'}]"
            f"{summary.margin_trend_pct_change:+.1f}%[/]",
            title="[cyan]Summary[/cyan]",
            border_style="cyan",
        )
    )

    # --- Top insight ---
    console.print(
        Panel(
            f"[italic]{summary.top_insight}[/italic]",
            title="[bold yellow]Top Insight[/bold yellow]",
            border_style="yellow",
        )
    )

    # --- Univariate results table ---
    if summary.univariate_insights:
        tbl = Table(title="Univariate Analysis", show_header=True, header_style="bold magenta")
        tbl.add_column("Metric", style="dim", min_width=25)
        tbl.add_column("Mean", justify="right")
        tbl.add_column("Median", justify="right")
        tbl.add_column("Skew", justify="right")
        for result in summary.univariate_insights:
            s = result.stats
            tbl.add_row(
                result.metric[:40],
                f"{s.get('mean', 0):,.0f}",
                f"{s.get('median', 0):,.0f}",
                f"{s.get('skew', 0):.2f}",
            )
        console.print(tbl)

    # --- Bivariate results table ---
    if summary.bivariate_insights:
        tbl = Table(title="Bivariate Analysis", show_header=True, header_style="bold magenta")
        tbl.add_column("X", style="dim", min_width=20)
        tbl.add_column("Y", style="dim", min_width=18)
        tbl.add_column("Corr (ρ/r)", justify="right")
        tbl.add_column("p-value", justify="right")
        tbl.add_column("Significant?", justify="center")
        for result in summary.bivariate_insights:
            sig_str = "[green]Yes[/green]" if result.is_significant else "[red]No[/red]"
            tbl.add_row(
                result.x_var[:22],
                result.y_var[:20],
                f"{result.correlation:.3f}",
                f"{result.p_value:.4f}",
                sig_str,
            )
        console.print(tbl)

    # --- Temporal results table ---
    if summary.temporal_insights:
        tbl = Table(title="Temporal Analysis", show_header=True, header_style="bold magenta")
        tbl.add_column("Metric", style="dim", min_width=25)
        tbl.add_column("Direction", justify="center")
        tbl.add_column("Magnitude (%)", justify="right")
        tbl.add_column("R²", justify="right")
        tbl.add_column("p-value", justify="right")
        for result in summary.temporal_insights:
            dir_colour = "red" if "declining" in result.trend_direction else "green"
            tbl.add_row(
                result.metric[:35],
                f"[{dir_colour}]{result.trend_direction}[/{dir_colour}]",
                f"{result.trend_magnitude:+.1f}",
                f"{result.trend_r_squared:.3f}",
                f"{result.trend_p_value:.4f}",
            )
        console.print(tbl)


if __name__ == "__main__":
    app()
