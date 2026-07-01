"""Generate strategic recommendations for NovaMart leadership.

Generates all 8 evidence-backed strategic recommendations from the data,
ranks them by RICE score, and saves the output to data/outputs/recommendations.json.

Usage:
    python scripts/generate_recommendations.py [OPTIONS]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Generate strategic recommendations for NovaMart leadership.")

# Ensure src is on PYTHONPATH when run as a script
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))


@app.command()
def main(
    outputs_dir: str = typer.Option(
        "data/outputs", help="Directory to write recommendation outputs"
    ),
    recs_output: str = typer.Option(
        "data/outputs/recommendations.json",
        help="Path to write the recommendations JSON",
    ),
    top_n: int = typer.Option(8, help="Number of top recommendations to include"),
) -> None:
    """Generate and rank strategic recommendations for NovaMart.

    Combines signals from:
    - Price elasticity coefficients (inelastic category opportunities)
    - Marketing channel ROI (digital vs TV in urban/rural formats)
    - Customer CLV and churn analysis (Silver tier rescue)
    - Store performance clustering (Cluster C portfolio rationalisation)
    - Discount rate statistical analysis (margin recovery)
    - Private label margin analysis
    """
    console.rule("[bold green]BCG X | NovaMart Recommendation Engine[/bold green]")

    # ── Load data ──────────────────────────────────────────────────────────────
    console.print("\n[cyan]Loading data...[/cyan]")
    from bcgx.data.loader import DataLoader
    from bcgx.recommendations.engine import RecommendationEngine
    from bcgx.recommendations.prioritizer import RecommendationPrioritizer

    loader = DataLoader()
    try:
        data = loader.load_all()
        console.print(f"  [green]✓[/green] Loaded {len(data)} datasets")
    except FileNotFoundError as e:
        console.print(f"  [red]✗ Data not found.[/red]\n  {e}")
        raise typer.Exit(1)

    # ── Generate recommendations ───────────────────────────────────────────────
    console.print("\n[cyan]Generating recommendations...[/cyan]")
    engine = RecommendationEngine(data_loader=loader)
    recs = engine.generate_all(data=data)
    console.print(f"  [green]✓[/green] Generated {len(recs)} recommendations")

    # ── Prioritise by RICE ─────────────────────────────────────────────────────
    console.print("\n[cyan]Prioritising by RICE score...[/cyan]")
    prioritizer = RecommendationPrioritizer()
    ranked = prioritizer.prioritize(recs)[:top_n]

    # ── Display results ────────────────────────────────────────────────────────
    table = Table(title="Strategic Recommendations (RICE-Ranked)", show_lines=True)
    table.add_column("Rank", style="bold", width=5)
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Title", width=48)
    table.add_column("Priority", width=10)
    table.add_column("Revenue $M", justify="right", width=11)
    table.add_column("Profit $M", justify="right", width=10)
    table.add_column("ROI", justify="right", width=7)
    table.add_column("RICE", justify="right", width=8)
    table.add_column("Timeline", width=14)

    priority_colours = {
        "Critical": "bold red",
        "High": "yellow",
        "Medium": "white",
        "Low": "dim",
    }

    for i, r in enumerate(ranked, 1):
        colour = priority_colours.get(r.priority.value, "white")
        table.add_row(
            str(i),
            r.id,
            r.title[:46],
            f"[{colour}]{r.priority.value}[/{colour}]",
            f"${r.expected_revenue_impact_usd/1e6:,.1f}M",
            f"${r.expected_profit_impact_usd/1e6:,.1f}M",
            f"{r.roi:.1f}x",
            f"{r.rice_score:.1f}",
            r.timeline.value,
        )

    console.print(table)

    total_rev = sum(r.expected_revenue_impact_usd for r in ranked)
    total_profit = sum(r.expected_profit_impact_usd for r in ranked)
    console.print(
        f"\n[bold green]Total addressable opportunity:[/bold green] "
        f"${total_rev/1e6:,.1f}M revenue · ${total_profit/1e6:,.1f}M profit"
    )

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_path = Path(recs_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_recommendations": len(ranked),
        "total_revenue_opportunity_usd": round(total_rev, 2),
        "total_profit_opportunity_usd": round(total_profit, 2),
        "recommendations": prioritizer.to_dict_list(ranked),
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    console.print(f"\n[green]✓[/green] Saved to [bold]{out_path}[/bold]")


if __name__ == "__main__":
    app()
