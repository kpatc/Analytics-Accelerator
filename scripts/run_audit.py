"""Run data quality audit across the NovaMart dataset.

Loads all parquet files from the raw data directory, runs DataAuditor on each,
prints a formatted report via AuditReporter, and saves a JSON report.

Usage:
    python scripts/run_audit.py [OPTIONS]

Options:
    --data-dir TEXT        Directory containing raw parquet files  [default: data/raw]
    --output-dir TEXT      Directory to write audit report  [default: data/outputs]
    --detail / --no-detail Print per-table column-level detail  [default: True]
    --fail-on-errors       Exit with code 1 if any rule violations exist  [default: False]
    --help                 Show this message and exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

# Ensure src/ is on the path when running as a script
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bcgx.data.loader import DataLoader
from bcgx.audit.quality import DataAuditor
from bcgx.audit.report import AuditReporter

console = Console()
app = typer.Typer(help="Run data quality audit on NovaMart datasets.", add_completion=False)


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


@app.command()
def main(
    data_dir: str = typer.Option("data/raw", help="Directory containing raw parquet files"),
    output_dir: str = typer.Option("data/outputs", help="Directory to write audit reports"),
    detail: bool = typer.Option(True, help="Print per-table column-level detail"),
    fail_on_errors: bool = typer.Option(
        False, "--fail-on-errors", help="Exit with code 1 if any business rule violations exist"
    ),
) -> None:
    """Run completeness, consistency, and distribution audits on NovaMart data.

    Generates:
    \b
    - Console report with summary table and per-table detail
    - data/outputs/audit_report.json  — Full JSON audit report
    """
    _configure_logging()
    console.print()
    console.print(
        Panel(
            "[bold cyan]NovaMart Data Quality Audit[/bold cyan]\n"
            f"[dim]data_dir={data_dir} | output_dir={output_dir}[/dim]",
            expand=False,
        )
    )

    # ── Load data ─────────────────────────────────────────────────────────
    console.print("\n  [bold]Loading datasets...[/bold]")
    try:
        loader = DataLoader(base_dir=data_dir)
        data = loader.load_all()
    except FileNotFoundError as exc:
        console.print(f"\n  [bold red]ERROR:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"  [green]Loaded {len(data)} tables[/green]")

    # ── Run audit ─────────────────────────────────────────────────────────
    console.print("\n  [bold]Running audit...[/bold]")
    auditor = DataAuditor()
    audits = auditor.audit_all(data)

    # ── Print report ──────────────────────────────────────────────────────
    reporter = AuditReporter(console=console)
    reporter.print_summary(audits)

    if detail:
        for name, audit in audits.items():
            reporter.print_table_detail(audit)

    # ── Save JSON report ──────────────────────────────────────────────────
    json_path = str(Path(output_dir) / "audit_report.json")
    reporter.save_json(audits, json_path)

    # ── Exit code ─────────────────────────────────────────────────────────
    total_violations = sum(len(a.business_rule_violations) for a in audits.values())
    if fail_on_errors and total_violations > 0:
        console.print(
            f"\n  [bold red]Exiting with code 1: {total_violations} business rule violation(s) found.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print()
    console.print(
        Panel(
            "[bold green]Audit complete[/bold green]\n"
            f"[dim]JSON report: {Path(json_path).resolve()}[/dim]",
            expand=False,
        )
    )


if __name__ == "__main__":
    app()
