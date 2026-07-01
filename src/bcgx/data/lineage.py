"""Data lineage tracking for NovaMart synthetic datasets.

Records metadata about each generated table — seed, timestamp, shape, version —
and persists them to JSON for audit trail and reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

GENERATOR_VERSION = "1.0.0"


@dataclass
class LineageRecord:
    """Metadata record for a single generated table."""

    table: str
    source: str  # e.g. "synthetic_generator"
    generated_at: str  # ISO 8601 timestamp
    seed: int
    row_count: int
    column_count: int
    generator_version: str


class DataLineage:
    """Records and persists data lineage for generated tables.

    Usage::

        lineage = DataLineage()
        records = []
        for name, df in data.items():
            records.append(lineage.record(name, df, seed=42))
        lineage.save(records, "data/raw/lineage.json")
        loaded = lineage.load("data/raw/lineage.json")
    """

    def record(self, table: str, df: pd.DataFrame, seed: int) -> LineageRecord:
        """Create a LineageRecord for a DataFrame.

        Args:
            table: Logical table name (e.g. "stores", "transactions").
            df: DataFrame that was generated.
            seed: Random seed used during generation.

        Returns:
            Populated LineageRecord.
        """
        rec = LineageRecord(
            table=table,
            source="synthetic_generator",
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            seed=seed,
            row_count=len(df),
            column_count=len(df.columns),
            generator_version=GENERATOR_VERSION,
        )
        logger.debug(
            f"Lineage recorded: {table} | {rec.row_count:,} rows × {rec.column_count} cols | "
            f"seed={seed} | version={GENERATOR_VERSION}"
        )
        return rec

    def save(self, records: list[LineageRecord], output_path: str) -> None:
        """Persist a list of LineageRecords to a JSON file.

        Creates the parent directory if it does not exist.

        Args:
            records: List of LineageRecord objects to serialise.
            output_path: Destination file path (will be overwritten if exists).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "record_count": len(records),
            "records": [asdict(r) for r in records],
        }
        out.write_text(json.dumps(payload, indent=2))
        logger.success(f"Lineage saved: {len(records)} records → {out}")

    def load(self, path: str) -> list[LineageRecord]:
        """Load LineageRecords from a JSON file previously saved by :meth:`save`.

        Args:
            path: Path to the JSON lineage file.

        Returns:
            List of LineageRecord objects.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Lineage file not found: {p}\n"
                "Run `make generate-data` to produce the raw data and lineage file."
            )
        raw = json.loads(p.read_text())
        records = [LineageRecord(**r) for r in raw["records"]]
        logger.info(f"Loaded {len(records)} lineage records from {p}")
        return records
