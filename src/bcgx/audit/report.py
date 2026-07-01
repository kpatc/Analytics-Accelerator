"""Audit report generator for NovaMart data quality results.

Produces formatted console output using `rich` and saves JSON reports.
Intended to be called after DataAuditor.audit_all() to display and persist results.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from bcgx.audit.quality import DataAuditor, TableAudit, ColumnProfile


_console = Console()


class AuditReporter:
    """Formats and displays audit results from DataAuditor.

    Usage::

        reporter = AuditReporter()
        reporter.print_summary(audits)
        reporter.print_table_detail(audits["transactions"])
        reporter.save_json(audits, "data/outputs/audit_report.json")
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or _console

    def print_summary(self, audits: dict[str, TableAudit]) -> None:
        """Print a high-level summary table of all audited tables.

        Shows: table name, row count, column count, duplicate %, high-null columns
        (>5% null), and columns with high outlier rates (>10%).

        Args:
            audits: Dict mapping table names to TableAudit results.
        """
        self._console.print()
        self._console.print(
            Panel(
                "[bold cyan]NovaMart Data Quality Audit — Summary[/bold cyan]",
                expand=False,
            )
        )

        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            expand=True,
        )
        table.add_column("Table", style="bold white", min_width=18)
        table.add_column("Rows", justify="right", style="cyan")
        table.add_column("Cols", justify="right", style="cyan")
        table.add_column("Dup Rows", justify="right")
        table.add_column("Dup %", justify="right")
        table.add_column("High-Null Cols (>5%)", style="yellow")
        table.add_column("High-Outlier Cols (>10%)", style="red")
        table.add_column("Rule Violations", justify="right")

        for name, audit in audits.items():
            high_null_cols = [
                f"{c.name} ({c.null_pct:.1f}%)"
                for c in audit.columns
                if c.null_pct > 5.0
            ]
            high_outlier_cols = [
                f"{c.name} ({c.outlier_pct:.1f}%)"
                for c in audit.columns
                if c.outlier_pct is not None and c.outlier_pct > 10.0
            ]

            dup_color = "red" if audit.duplicate_row_pct > 1.0 else "green"
            violations_color = "red" if audit.business_rule_violations else "green"

            table.add_row(
                name,
                f"{audit.row_count:,}",
                str(audit.column_count),
                f"[{dup_color}]{audit.duplicate_row_count:,}[/{dup_color}]",
                f"[{dup_color}]{audit.duplicate_row_pct:.2f}%[/{dup_color}]",
                ", ".join(high_null_cols) if high_null_cols else "[green]none[/green]",
                ", ".join(high_outlier_cols) if high_outlier_cols else "[green]none[/green]",
                f"[{violations_color}]{len(audit.business_rule_violations)}[/{violations_color}]",
            )

        self._console.print(table)
        self._console.print()

        # Summary stats
        total_rows = sum(a.row_count for a in audits.values())
        total_violations = sum(len(a.business_rule_violations) for a in audits.values())
        self._console.print(
            f"  [dim]Total tables: {len(audits)} | "
            f"Total rows: {total_rows:,} | "
            f"Rule violations: {total_violations}[/dim]"
        )
        self._console.print()

    def print_table_detail(self, audit: TableAudit) -> None:
        """Print a detailed column-level profile for a single table.

        Shows dtype, null stats, unique stats, and numeric statistics for each column.

        Args:
            audit: TableAudit to display.
        """
        self._console.print()
        self._console.print(
            Panel(
                f"[bold cyan]Table Detail: [white]{audit.table_name}[/white][/bold cyan]  "
                f"[dim]({audit.row_count:,} rows × {audit.column_count} cols | "
                f"{audit.duplicate_row_count:,} duplicate rows / {audit.duplicate_row_pct:.2f}%)[/dim]",
                expand=False,
            )
        )

        # Column profile table
        col_table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        col_table.add_column("Column", style="bold white", min_width=22)
        col_table.add_column("Dtype", min_width=10)
        col_table.add_column("Nulls", justify="right")
        col_table.add_column("Null%", justify="right")
        col_table.add_column("Unique", justify="right")
        col_table.add_column("Unique%", justify="right")
        col_table.add_column("Mean", justify="right")
        col_table.add_column("Std", justify="right")
        col_table.add_column("Min", justify="right")
        col_table.add_column("P50", justify="right")
        col_table.add_column("Max", justify="right")
        col_table.add_column("Outliers", justify="right")

        for col in audit.columns:
            null_style = "red" if col.null_pct > 5.0 else ""
            outlier_style = "red" if (col.outlier_pct or 0) > 10.0 else ""

            col_table.add_row(
                col.name,
                col.dtype,
                f"[{null_style}]{col.null_count:,}[/{null_style}]" if null_style else f"{col.null_count:,}",
                f"[{null_style}]{col.null_pct:.2f}%[/{null_style}]" if null_style else f"{col.null_pct:.2f}%",
                f"{col.unique_count:,}",
                f"{col.unique_pct:.2f}%",
                f"{col.mean:.2f}" if col.mean is not None else "—",
                f"{col.std:.2f}" if col.std is not None else "—",
                f"{col.min:.2f}" if col.min is not None else "—",
                f"{col.p50:.2f}" if col.p50 is not None else "—",
                f"{col.max:.2f}" if col.max is not None else "—",
                (
                    f"[{outlier_style}]{col.outlier_count:,} ({col.outlier_pct:.1f}%)[/{outlier_style}]"
                    if outlier_style and col.outlier_count is not None
                    else (f"{col.outlier_count:,} ({col.outlier_pct:.1f}%)" if col.outlier_count is not None else "—")
                ),
            )

        self._console.print(col_table)

        # Business rule violations
        if audit.business_rule_violations:
            self._console.print(
                f"\n  [bold red]Business Rule Violations ({len(audit.business_rule_violations)}):[/bold red]"
            )
            for v in audit.business_rule_violations:
                self._console.print(f"    [red]✗[/red] {v}")
        else:
            self._console.print("  [green]✓ No business rule violations[/green]")

        self._console.print()

    def save_json(self, audits: dict[str, TableAudit], output_path: str) -> None:
        """Serialise all audit results to a JSON file.

        Args:
            audits: Dict mapping table names to TableAudit results.
            output_path: File path to write the JSON report.
        """
        auditor = DataAuditor()
        report: dict = {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "tables": {name: auditor.to_dict(audit) for name, audit in audits.items()},
            "summary": {
                "total_tables": len(audits),
                "total_rows": sum(a.row_count for a in audits.values()),
                "total_rule_violations": sum(
                    len(a.business_rule_violations) for a in audits.values()
                ),
            },
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        logger.success(f"Audit report saved to {out_path}")
        self._console.print(f"  [green]Audit JSON saved to:[/green] [dim]{out_path}[/dim]")
