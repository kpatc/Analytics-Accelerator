"""Simple file-based feature store for NovaMart analytics.

Provides a consistent interface for saving and loading feature matrices as parquet files,
with a JSON manifest tracking table schemas, shapes, and timestamps.

Usage::

    from bcgx.features.feature_store import FeatureStore

    fs = FeatureStore("data/features")
    fs.save_store_features(store_df)
    fs.save_customer_features(customer_df)

    store_df = fs.load_store_features()
    manifest = fs.get_feature_manifest()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger


class FeatureStore:
    """Simple file-based feature store backed by parquet files.

    All feature tables are stored in BASE_DIR as parquet files.
    A manifest.json tracks schema metadata (columns, shape, last saved timestamp).

    Args:
        base_dir: Directory for feature parquet files and manifest. Created if absent.
    """

    _TABLE_NAMES: dict[str, str] = {
        "store": "store_features.parquet",
        "customer": "customer_features.parquet",
        "product": "product_features.parquet",
    }
    _MANIFEST_FILE: str = "manifest.json"

    def __init__(self, base_dir: str = "data/features") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FeatureStore initialised at {self._base_dir}")

    # ------------------------------------------------------------------ #
    # Save methods                                                         #
    # ------------------------------------------------------------------ #

    def save_store_features(self, df: pd.DataFrame) -> None:
        """Persist the store feature matrix to parquet.

        Args:
            df: Store feature DataFrame (one row per store, indexed by store_id).
        """
        self._save("store", df)

    def save_customer_features(self, df: pd.DataFrame) -> None:
        """Persist the customer feature matrix to parquet.

        Args:
            df: Customer feature DataFrame (one row per customer, indexed by customer_id).
        """
        self._save("customer", df)

    def save_product_features(self, df: pd.DataFrame) -> None:
        """Persist the product feature matrix to parquet.

        Args:
            df: Product feature DataFrame (one row per product, indexed by product_id).
        """
        self._save("product", df)

    # ------------------------------------------------------------------ #
    # Load methods                                                         #
    # ------------------------------------------------------------------ #

    def load_store_features(self) -> pd.DataFrame:
        """Load the store feature matrix.

        Returns:
            Store feature DataFrame.

        Raises:
            FileNotFoundError: If the store feature file does not exist.
        """
        return self._load("store")

    def load_customer_features(self) -> pd.DataFrame:
        """Load the customer feature matrix.

        Returns:
            Customer feature DataFrame.

        Raises:
            FileNotFoundError: If the customer feature file does not exist.
        """
        return self._load("customer")

    def load_product_features(self) -> pd.DataFrame:
        """Load the product feature matrix.

        Returns:
            Product feature DataFrame.

        Raises:
            FileNotFoundError: If the product feature file does not exist.
        """
        return self._load("product")

    # ------------------------------------------------------------------ #
    # Manifest                                                             #
    # ------------------------------------------------------------------ #

    def get_feature_manifest(self) -> dict[str, dict]:  # type: ignore[type-arg]
        """Return the feature manifest describing all stored tables.

        Returns:
            Dict mapping table names to metadata dicts with keys:
            'columns', 'shape', 'saved_at', 'file'.
        """
        manifest_path = self._base_dir / self._MANIFEST_FILE
        if not manifest_path.exists():
            logger.warning("Manifest not found — returning empty manifest")
            return {}

        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def save_manifest(self) -> None:
        """Rebuild and write the manifest from the current parquet files on disk."""
        manifest: dict[str, dict] = {}  # type: ignore[type-arg]
        for table_key, filename in self._TABLE_NAMES.items():
            path = self._base_dir / filename
            if path.exists():
                df = pd.read_parquet(path)
                manifest[table_key] = {
                    "file": filename,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "saved_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
        manifest_path = self._base_dir / self._MANIFEST_FILE
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Feature manifest saved to {manifest_path}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _save(self, table_key: str, df: pd.DataFrame) -> None:
        """Save a DataFrame as parquet and update the manifest."""
        filename = self._TABLE_NAMES[table_key]
        path = self._base_dir / filename
        df.to_parquet(path, index=True)
        logger.info(f"Saved {table_key} features → {path} ({df.shape[0]:,} rows × {df.shape[1]} cols)")
        self._update_manifest_entry(table_key, df, filename)

    def _load(self, table_key: str) -> pd.DataFrame:
        """Load a parquet file by table key."""
        filename = self._TABLE_NAMES[table_key]
        path = self._base_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Feature table '{table_key}' not found at {path}.\n"
                "Run the feature engineering pipeline first:\n"
                "    python scripts/run_eda.py"
            )
        df = pd.read_parquet(path)
        logger.debug(f"Loaded {table_key} features from {path}: {df.shape}")
        return df

    def _update_manifest_entry(self, table_key: str, df: pd.DataFrame, filename: str) -> None:
        """Update a single entry in the manifest JSON without overwriting other entries."""
        manifest = self.get_feature_manifest()
        manifest[table_key] = {
            "file": filename,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = self._base_dir / self._MANIFEST_FILE
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
