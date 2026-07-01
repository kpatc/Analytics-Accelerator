"""Generate synthetic NovaMart retail data.

CLI entry point for the NovaMart data generation pipeline.
Produces seven parquet datasets covering stores, customers, products,
transactions, marketing spend, inventory, and store costs.

Usage:
    python scripts/generate_data.py [OPTIONS]

Options:
    --seed INTEGER        Random seed for reproducibility  [default: 42]
    --output-dir TEXT     Directory to write raw data files  [default: data/raw]
    --validate / --no-validate
                          Run DataValidator after generation  [default: True]
    --verbose / --no-verbose
                          Show DEBUG-level log messages  [default: False]
    --help                Show this message and exit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Ensure src/ is on the path when running as a script
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bcgx.data.generator import generate_all, generate_stores, generate_customers, generate_products
from bcgx.data.validator import DataValidator, ValidationResult
from bcgx.data.lineage import DataLineage

console = Console()
app = typer.Typer(help="Generate synthetic NovaMart retail data.", add_completion=False)


def _configure_logging(verbose: bool) -> None:
    """Set up loguru logging level."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


def _print_validation_summary(results: dict[str, ValidationResult]) -> None:
    """Print validation results as a rich table."""
    console.print()
    table = Table(
        title="Validation Results",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        expand=False,
    )
    table.add_column("Table", style="bold white", min_width=18)
    table.add_column("Status", justify="center")
    table.add_column("Checks Passed", justify="right")
    table.add_column("Checks Failed", justify="right")

    all_passed = True
    for name, result in results.items():
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        if not result.passed:
            all_passed = False
        table.add_row(
            name,
            status,
            str(result.n_passed),
            f"[red]{result.n_failed}[/red]" if result.n_failed else str(result.n_failed),
        )

    console.print(table)

    if not all_passed:
        console.print("\n  [yellow]Some validation checks failed. Review the log output above.[/yellow]")
    else:
        console.print("\n  [green]All validation checks passed.[/green]")


@app.command()
def main(
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    output_dir: str = typer.Option("data/raw", help="Output directory for raw parquet data"),
    validate: bool = typer.Option(True, help="Run DataValidator after generation"),
    verbose: bool = typer.Option(False, help="Show DEBUG-level log messages"),
) -> None:
    """Generate the full suite of synthetic NovaMart retail datasets.

    Produces the following parquet files in OUTPUT_DIR:
    \b
    - stores.parquet        — Store master (800 stores)
    - customers.parquet     — Customer master (500,000 customers)
    - products.parquet      — Product catalogue (5,000 products)
    - transactions.parquet  — Transaction-level sales (~500K rows / 36 months)
    - marketing_spend.parquet — Monthly marketing spend by channel
    - inventory.parquet     — Monthly inventory metrics by store × category
    - store_costs.parquet   — Monthly store operating costs
    """
    _configure_logging(verbose)
    console.print()
    console.print(
        Panel(
            "[bold cyan]NovaMart Synthetic Data Generator[/bold cyan]\n"
            f"[dim]seed={seed} | output_dir={output_dir}[/dim]",
            expand=False,
        )
    )

    # ── Generation ────────────────────────────────────────────────────────
    t_gen_start = time.perf_counter()
    data = generate_all(seed=seed, output_dir=output_dir)
    t_gen = time.perf_counter() - t_gen_start

    # ── Display generation summary ────────────────────────────────────────
    gen_table = Table(
        title=f"Generated Datasets  [dim](in {t_gen:.1f}s)[/dim]",
        show_header=True,
        header_style="bold blue",
        box=box.ROUNDED,
        expand=False,
    )
    gen_table.add_column("Table", style="bold white", min_width=20)
    gen_table.add_column("Rows", justify="right", style="cyan")
    gen_table.add_column("Columns", justify="right", style="cyan")
    gen_table.add_column("File", style="dim")

    total_rows = 0
    for name, df in data.items():
        total_rows += len(df)
        gen_table.add_row(
            name,
            f"{len(df):,}",
            str(len(df.columns)),
            f"{output_dir}/{name}.parquet",
        )

    console.print()
    console.print(gen_table)
    console.print(f"\n  [dim]Total rows across all tables: {total_rows:,}[/dim]")

    # ── Lineage ───────────────────────────────────────────────────────────
    t_lineage_start = time.perf_counter()
    lineage = DataLineage()
    records = [lineage.record(name, df, seed) for name, df in data.items()]
    lineage_path = str(Path(output_dir) / "lineage.json")
    lineage.save(records, lineage_path)
    t_lineage = time.perf_counter() - t_lineage_start
    console.print(f"  [dim]Lineage recorded in {t_lineage:.2f}s → {lineage_path}[/dim]")

    # ── Validation ────────────────────────────────────────────────────────
    if validate:
        console.print("\n  [bold]Running data validation...[/bold]")
        t_val_start = time.perf_counter()
        validator = DataValidator()
        val_results = validator.validate_all(
            {k: v for k, v in data.items() if k in {"stores", "customers", "products", "transactions"}}
        )
        t_val = time.perf_counter() - t_val_start
        _print_validation_summary(val_results)
        console.print(f"  [dim]Validation completed in {t_val:.2f}s[/dim]")
    else:
        console.print("\n  [dim]Validation skipped (--no-validate)[/dim]")

    # ── Done ──────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold green]Data generation complete[/bold green]  "
            f"[dim]Total time: {time.perf_counter() - t_gen_start:.1f}s[/dim]\n"
            f"Files saved to: [cyan]{Path(output_dir).resolve()}[/cyan]",
            expand=False,
        )
    )


if __name__ == "__main__":
    app()
