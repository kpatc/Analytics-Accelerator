"""Data loading module for NovaMart raw parquet datasets.

Provides a typed interface to load each dataset from the raw data directory.
Raises clear errors with actionable instructions when files are missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


class DataLoader:
    """Loads NovaMart parquet files from a base directory.

    Args:
        base_dir: Directory containing the raw parquet files (default: "data/raw").

    Usage::

        loader = DataLoader("data/raw")
        stores = loader.load_stores()
        data = loader.load_all()
    """

    # Maps logical table names to their parquet filenames
    _FILE_MAP: dict[str, str] = {
        "stores": "stores.parquet",
        "customers": "customers.parquet",
        "products": "products.parquet",
        "transactions": "transactions.parquet",
        "marketing_spend": "marketing_spend.parquet",
        "inventory": "inventory.parquet",
        "store_costs": "store_costs.parquet",
    }

    def __init__(self, base_dir: str = "data/raw") -> None:
        self._base_dir = Path(base_dir)

    def _load(self, table_name: str) -> pd.DataFrame:
        """Load a single parquet file by table name.

        Args:
            table_name: Logical table name key from _FILE_MAP.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
        """
        filename = self._FILE_MAP[table_name]
        path = self._base_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n\n"
                f"The '{table_name}' dataset has not been generated yet.\n"
                f"Run the following command to generate all datasets:\n\n"
                f"    make generate-data\n\n"
                f"Or run the script directly:\n\n"
                f"    python scripts/generate_data.py\n"
            )

        logger.debug(f"Loading {table_name} from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {table_name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    def load_stores(self) -> pd.DataFrame:
        """Load the stores master table.

        Returns:
            DataFrame with one row per store (800 stores).
        """
        return self._load("stores")

    def load_customers(self) -> pd.DataFrame:
        """Load the customer master table.

        Returns:
            DataFrame with one row per customer (500,000 customers).
        """
        return self._load("customers")

    def load_products(self) -> pd.DataFrame:
        """Load the product catalogue.

        Returns:
            DataFrame with one row per product (5,000 products).
        """
        return self._load("products")

    def load_transactions(self) -> pd.DataFrame:
        """Load the transaction-level sales data.

        Returns:
            DataFrame with ~500K transaction rows over 36 months.
        """
        return self._load("transactions")

    def load_marketing_spend(self) -> pd.DataFrame:
        """Load the monthly marketing spend data.

        Returns:
            DataFrame with one row per store × month × channel.
        """
        return self._load("marketing_spend")

    def load_inventory(self) -> pd.DataFrame:
        """Load the monthly inventory data.

        Returns:
            DataFrame with one row per store × category × month.
        """
        return self._load("inventory")

    def load_store_costs(self) -> pd.DataFrame:
        """Load the monthly store operating costs.

        Returns:
            DataFrame with one row per store × month.
        """
        return self._load("store_costs")

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available datasets.

        Returns:
            Dict mapping table names to DataFrames.

        Raises:
            FileNotFoundError: If any required parquet file is missing.
        """
        logger.info(f"Loading all NovaMart datasets from {self._base_dir}")
        data: dict[str, pd.DataFrame] = {}
        for table_name in self._FILE_MAP:
            data[table_name] = self._load(table_name)
        logger.success(f"All {len(data)} datasets loaded successfully")
        return data
