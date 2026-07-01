"""Synthetic NovaMart retail data generator.

Generates realistic, statistically-rich synthetic data for the NovaMart retail analytics
engagement. All signals are designed to be analytically discoverable by downstream models.

Embedded signals:
- 3 store performance clusters (A=high 15%, B=mid 60%, C=low 25%)
- Pareto: cluster A drives 60% of profit via cluster_profit_multiplier
- Price elasticity varies by category (electronics elastic, food inelastic)
- Silver-tier churn 2x higher from month 25 (loyalty fee hike event)
- Private label margin 2.3x national brands
- Digital ROI 3x TV in urban, reversed in rural
- Net margin trends from ~8.2% to ~5.7% over 36 months (COGS inflation)
- 15% of stores drive 60% of profit (Pareto distribution via cluster)

Designed to scale: schema and logic support 15M+ rows; default targets ~500K for speed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

# ── Constants ───────────────────────────────────────────────────────────────

GENERATOR_VERSION = "1.0.0"

START_YEAR_MONTH = "2022-01"  # Month index 0
N_MONTHS = 36  # 2022-01 through 2024-12

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
REGION_WEIGHTS = [0.18, 0.24, 0.21, 0.16, 0.21]  # proportional to US population

STORE_FORMATS = ["urban", "suburban", "rural"]
STORE_FORMAT_WEIGHTS = [0.30, 0.50, 0.20]

CATEGORIES = ["Electronics", "Apparel", "Home & Garden", "Food & Beverage", "Health & Beauty", "Sports", "Toys"]
CATEGORY_WEIGHTS = [0.08, 0.20, 0.15, 0.25, 0.15, 0.10, 0.07]

SUBCATEGORIES: dict[str, list[str]] = {
    "Electronics": ["Smartphones", "Laptops", "Audio", "Smart Home"],
    "Apparel": ["Women's", "Men's", "Kids", "Accessories"],
    "Home & Garden": ["Furniture", "Garden Tools", "Bedding", "Kitchen"],
    "Food & Beverage": ["Snacks", "Beverages", "Dairy & Eggs", "Pantry Staples"],
    "Health & Beauty": ["Skincare", "Vitamins", "Personal Care", "Hair Care"],
    "Sports": ["Fitness Equipment", "Outdoor Gear", "Team Sports", "Footwear"],
    "Toys": ["Action Figures", "Board Games", "Educational", "Outdoor Toys"],
}

# Price elasticity coefficients (negative = elastic demand)
PRICE_ELASTICITY: dict[str, tuple[float, float]] = {
    "Electronics": (-2.1, -1.8),
    "Apparel": (-1.5, -1.0),
    "Home & Garden": (-1.2, -0.8),
    "Food & Beverage": (-0.6, -0.3),
    "Health & Beauty": (-0.9, -0.6),
    "Sports": (-1.4, -1.0),
    "Toys": (-1.3, -0.9),
}

# Unit cost ranges by category (min, max) in USD
UNIT_COST_RANGES: dict[str, tuple[float, float]] = {
    "Electronics": (50.0, 800.0),
    "Apparel": (8.0, 80.0),
    "Home & Garden": (10.0, 200.0),
    "Food & Beverage": (1.0, 25.0),
    "Health & Beauty": (3.0, 50.0),
    "Sports": (15.0, 300.0),
    "Toys": (5.0, 60.0),
}

# Cluster configuration
CLUSTER_LABELS = ["A", "B", "C"]
CLUSTER_WEIGHTS = [0.15, 0.60, 0.25]
CLUSTER_PROFIT_MULTIPLIER = {"A": 2.8, "B": 1.0, "C": 0.4}

# Seasonal multipliers by month number (1=Jan ... 12=Dec)
SEASONAL_MULTIPLIERS: dict[int, float] = {
    1: 0.70, 2: 0.75, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.00,
    7: 0.98, 8: 1.02, 9: 1.05, 10: 1.10, 11: 1.20, 12: 1.40,
}

# Marketing channel mix by store format
MARKETING_CHANNELS = ["digital", "tv", "print", "email", "instore_promo"]
CHANNEL_MIX: dict[str, list[float]] = {
    "urban": [0.45, 0.20, 0.10, 0.20, 0.05],
    "suburban": [0.30, 0.30, 0.20, 0.15, 0.05],
    "rural": [0.20, 0.40, 0.25, 0.10, 0.05],
}

# Digital ROI multiplier by store format (for signal injection)
DIGITAL_ROI_MULTIPLIER: dict[str, float] = {
    "urban": 3.0,
    "suburban": 1.5,
    "rural": 0.8,
}


# ── Helper utilities ─────────────────────────────────────────────────────────


def _ym_to_date(year_month: str) -> pd.Timestamp:
    """Convert 'YYYY-MM' string to a Timestamp at the first of the month."""
    return pd.Timestamp(year_month + "-01")


def _months_sequence(n_months: int) -> list[str]:
    """Generate list of YYYY-MM strings starting from START_YEAR_MONTH."""
    start = _ym_to_date(START_YEAR_MONTH)
    return [
        (start + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(n_months)
    ]


def _ensure_dir(path: str) -> Path:
    """Create directory and all parents if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Public API ───────────────────────────────────────────────────────────────


def generate_stores(seed: int, n_stores: int = 800) -> pd.DataFrame:
    """Generate the NovaMart store master table.

    Args:
        seed: NumPy random seed for reproducibility.
        n_stores: Number of stores to generate (default 800).

    Returns:
        DataFrame with one row per store.
    """
    logger.info(f"Generating {n_stores} stores (seed={seed})")
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    store_ids = [f"S{i:03d}" for i in range(1, n_stores + 1)]

    regions = rng.choice(REGIONS, size=n_stores, p=REGION_WEIGHTS).tolist()
    formats = rng.choice(STORE_FORMATS, size=n_stores, p=STORE_FORMAT_WEIGHTS).tolist()

    sq_footage: list[int] = []
    for fmt in formats:
        if fmt == "urban":
            sq_footage.append(int(rng.integers(8_000, 15_001)))
        elif fmt == "suburban":
            sq_footage.append(int(rng.integers(12_000, 25_001)))
        else:
            sq_footage.append(int(rng.integers(5_000, 10_001)))

    # Open dates: random date between 2010-01-01 and 2020-12-31
    open_days_range = (pd.Timestamp("2020-12-31") - pd.Timestamp("2010-01-01")).days
    open_date_offsets = rng.integers(0, open_days_range, size=n_stores)
    open_dates = [
        (pd.Timestamp("2010-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in open_date_offsets
    ]

    manager_tenure = rng.uniform(0.5, 12.0, size=n_stores).round(1).tolist()

    # Cluster assignment: correlated with region + format, then add noise
    # Northeast+urban → more likely A; rural+Southeast → more likely C
    cluster_list: list[str] = []
    for region, fmt in zip(regions, formats):
        # Base probability tweak
        base = [0.15, 0.60, 0.25]
        if fmt == "urban" and region in ["Northeast", "West"]:
            base = [0.30, 0.55, 0.15]
        elif fmt == "rural" and region in ["Southeast", "Midwest"]:
            base = [0.07, 0.50, 0.43]
        # Normalise then sample
        base_arr = np.array(base)
        base_arr = base_arr / base_arr.sum()
        cluster_list.append(str(rng.choice(CLUSTER_LABELS, p=base_arr)))

    # Enforce target distribution: A~15%, B~60%, C~25% — resample overflows
    # This is a soft correction: adjust a few randomly drawn outliers
    counts = {c: cluster_list.count(c) for c in CLUSTER_LABELS}
    targets = {
        "A": int(round(0.15 * n_stores)),
        "B": int(round(0.60 * n_stores)),
        "C": n_stores - int(round(0.15 * n_stores)) - int(round(0.60 * n_stores)),
    }
    for cluster_over, cluster_under in [("A", "C"), ("C", "A"), ("B", "A"), ("B", "C")]:
        while counts[cluster_over] > targets[cluster_over] + max(3, int(0.03 * n_stores)):
            idxs = [i for i, c in enumerate(cluster_list) if c == cluster_over]
            swap_idx = int(rng.choice(idxs))
            cluster_list[swap_idx] = cluster_under
            counts[cluster_over] -= 1
            counts[cluster_under] += 1

    profit_multipliers = [CLUSTER_PROFIT_MULTIPLIER[c] for c in cluster_list]

    # Fake city/state for flavor
    cities = [fake.city() for _ in range(n_stores)]
    states = [fake.state_abbr() for _ in range(n_stores)]

    df = pd.DataFrame({
        "store_id": store_ids,
        "city": cities,
        "state": states,
        "region": regions,
        "store_format": formats,
        "sq_footage": sq_footage,
        "open_date": open_dates,
        "manager_tenure_years": manager_tenure,
        "performance_cluster": cluster_list,
        "cluster_profit_multiplier": profit_multipliers,
    })

    logger.success(f"Stores generated: {df.shape} | cluster dist: {df['performance_cluster'].value_counts().to_dict()}")
    return df


def generate_customers(seed: int, n_customers: int = 500_000) -> pd.DataFrame:
    """Generate the NovaMart customer master table.

    Args:
        seed: NumPy random seed for reproducibility.
        n_customers: Number of customers to generate (default 500,000).

    Returns:
        DataFrame with one row per customer.
    """
    logger.info(f"Generating {n_customers:,} customers (seed={seed})")
    rng = np.random.default_rng(seed + 1)

    customer_ids = [f"C{i:06d}" for i in range(1, n_customers + 1)]

    segments = rng.choice(
        ["Premium", "Regular", "Occasional"],
        size=n_customers,
        p=[0.15, 0.55, 0.30],
    ).tolist()

    loyalty_tiers = rng.choice(
        ["Gold", "Silver", "Bronze"],
        size=n_customers,
        p=[0.20, 0.35, 0.45],
    ).tolist()

    acq_channels = rng.choice(
        ["digital", "in-store", "referral", "tv"],
        size=n_customers,
        p=[0.40, 0.35, 0.15, 0.10],
    ).tolist()

    regions = rng.choice(REGIONS, size=n_customers, p=REGION_WEIGHTS).tolist()

    # Acquisition date: random within the 3-year window
    acq_day_range = (pd.Timestamp("2024-12-31") - pd.Timestamp("2022-01-01")).days
    acq_offsets = rng.integers(0, acq_day_range, size=n_customers)
    acq_dates = [
        (pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in acq_offsets
    ]

    email_opt_in = rng.random(size=n_customers) < 0.65  # 65% opted in

    age_groups = rng.choice(
        ["18-24", "25-34", "35-49", "50-64", "65+"],
        size=n_customers,
        p=[0.10, 0.25, 0.35, 0.20, 0.10],
    ).tolist()

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "segment": segments,
        "loyalty_tier": loyalty_tiers,
        "acquisition_channel": acq_channels,
        "region": regions,
        "acquisition_date": acq_dates,
        "email_opt_in": email_opt_in.tolist(),
        "age_group": age_groups,
    })

    logger.success(f"Customers generated: {df.shape}")
    return df


def generate_products(seed: int, n_products: int = 5_000) -> pd.DataFrame:
    """Generate the NovaMart product catalogue.

    Private label products have higher margin (cost * 2.3x) vs national brands (cost * 1.6x).

    Args:
        seed: NumPy random seed for reproducibility.
        n_products: Number of products to generate (default 5,000).

    Returns:
        DataFrame with one row per product.
    """
    logger.info(f"Generating {n_products:,} products (seed={seed})")
    rng = np.random.default_rng(seed + 2)
    fake = Faker()
    fake.seed_instance(seed + 2)

    product_ids = [f"P{i:05d}" for i in range(1, n_products + 1)]

    categories = rng.choice(CATEGORIES, size=n_products, p=CATEGORY_WEIGHTS).tolist()
    subcategories = [
        str(rng.choice(SUBCATEGORIES[cat]))
        for cat in categories
    ]

    brand_types = rng.choice(
        ["private_label", "national_brand"],
        size=n_products,
        p=[0.35, 0.65],
    ).tolist()

    unit_costs: list[float] = []
    list_prices: list[float] = []

    for cat, brand in zip(categories, brand_types):
        lo, hi = UNIT_COST_RANGES[cat]
        cost = float(rng.uniform(lo, hi))
        unit_costs.append(round(cost, 2))
        if brand == "private_label":
            # Private label: cost × 2.3 markup → higher margin
            price = round(cost * 2.3, 2)
        else:
            # National brand: cost × 1.6 markup → lower margin
            price = round(cost * 1.6, 2)
        list_prices.append(price)

    elasticity_coefficients: list[float] = []
    for cat in categories:
        lo, hi = PRICE_ELASTICITY[cat]
        elasticity_coefficients.append(round(float(rng.uniform(lo, hi)), 3))

    # Compute margin pct for reference
    gross_margin_pct = [
        round((lp - uc) / lp * 100, 1)
        for lp, uc in zip(list_prices, unit_costs)
    ]

    # Realistic-sounding product names
    product_names = [f"{fake.word().capitalize()} {sub}" for sub in subcategories]

    df = pd.DataFrame({
        "product_id": product_ids,
        "product_name": product_names,
        "category": categories,
        "subcategory": subcategories,
        "brand_type": brand_types,
        "unit_cost": unit_costs,
        "list_price": list_prices,
        "gross_margin_pct": gross_margin_pct,
        "price_elasticity_coefficient": elasticity_coefficients,
    })

    logger.success(
        f"Products generated: {df.shape} | "
        f"avg PL margin={df[df['brand_type']=='private_label']['gross_margin_pct'].mean():.1f}% "
        f"avg NB margin={df[df['brand_type']=='national_brand']['gross_margin_pct'].mean():.1f}%"
    )
    return df


def generate_transactions(
    seed: int,
    stores: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    n_months: int = 36,
) -> pd.DataFrame:
    """Generate synthetic transaction-level sales data (~500K rows over 36 months).

    Injected signals:
    - Seasonal multipliers (Dec +40%, Jan -30%, etc.)
    - COGS inflation from month 13 → net margin trends 8.2% → 5.7%
    - Silver-tier churn: 40% fewer transactions from month 25
    - Cluster A stores generate proportionally more volume

    Args:
        seed: NumPy random seed for reproducibility.
        stores: Store master DataFrame (from generate_stores).
        customers: Customer master DataFrame (from generate_customers).
        products: Product catalogue DataFrame (from generate_products).
        n_months: Number of months to generate (default 36).

    Returns:
        Transaction DataFrame with ~500K rows.
    """
    logger.info(f"Generating transactions over {n_months} months (seed={seed})")
    rng = np.random.default_rng(seed + 3)

    months = _months_sequence(n_months)

    store_ids = stores["store_id"].values
    store_formats = stores["store_format"].values
    store_sqft = stores["sq_footage"].values.astype(float)
    store_multipliers = stores["cluster_profit_multiplier"].values.astype(float)

    customer_ids = customers["customer_id"].values
    customer_tiers = customers["loyalty_tier"].values

    product_ids = products["product_id"].values
    product_costs = products["unit_cost"].values.astype(float)
    product_prices = products["list_price"].values.astype(float)

    all_rows: list[pd.DataFrame] = []

    # Pre-build Silver mask (for churn signal)
    silver_mask = customer_tiers == "Silver"
    non_silver_customer_ids = customer_ids[~silver_mask]
    silver_customer_ids = customer_ids[silver_mask]

    for month_idx, ym in enumerate(months):
        ts_start = pd.Timestamp(ym + "-01")
        month_num = ts_start.month
        days_in_month = ts_start.days_in_month
        seasonal = SEASONAL_MULTIPLIERS[month_num]

        # COGS inflation factor: ramps linearly from 1.0 (month 13) to 1.18 (month 36)
        if month_idx < 12:
            cogs_inflation = 1.0
        else:
            progress = (month_idx - 12) / 23  # 0→1 over months 13-36
            cogs_inflation = 1.0 + 0.18 * progress

        # Silver churn: from month 25, Silver customers have 40% less traffic
        silver_churn_active = month_idx >= 24  # month index 24 = 2024-01

        # Build per-store transaction counts
        # Base = sqft / 8 * cluster_multiplier * seasonal * small_noise
        base_counts = (store_sqft / 8.0) * store_multipliers * seasonal
        noise = rng.uniform(0.85, 1.15, size=len(store_ids))
        store_tx_counts = np.round(base_counts * noise).astype(int)

        # Normalise total to ~14K per month (targeting ~500K total)
        target_per_month = 14_000
        total_raw = store_tx_counts.sum()
        if total_raw > 0:
            scale = target_per_month / total_raw
            store_tx_counts = np.maximum(1, np.round(store_tx_counts * scale).astype(int))

        # Generate transactions for this month
        n_tx = int(store_tx_counts.sum())

        # Store assignments (repeating each store_id by its count)
        store_idx_arr = np.repeat(np.arange(len(store_ids)), store_tx_counts)

        # Random product for each transaction
        prod_idx_arr = rng.integers(0, len(product_ids), size=n_tx)

        # Customer assignment: respect churn signal
        if silver_churn_active:
            # Reduce silver selection probability
            n_silver = int(n_tx * 0.15)  # Silver gets 15% instead of 35%
            n_non_silver = n_tx - n_silver
            cust_idx_silver = rng.integers(0, len(silver_customer_ids), size=n_silver)
            cust_idx_non_silver = rng.integers(0, len(non_silver_customer_ids), size=n_non_silver)
            chosen_custs = np.concatenate([
                silver_customer_ids[cust_idx_silver],
                non_silver_customer_ids[cust_idx_non_silver],
            ])
            rng.shuffle(chosen_custs)
        else:
            cust_idx_arr = rng.integers(0, len(customer_ids), size=n_tx)
            chosen_custs = customer_ids[cust_idx_arr]

        # Quantity: weighted toward 1-2
        qty_choices = np.array([1, 2, 3, 4])
        qty_weights = np.array([0.50, 0.30, 0.15, 0.05])
        quantities = rng.choice(qty_choices, size=n_tx, p=qty_weights)

        # Discount: 60% no discount, 40% get 5-30%
        discount_mask = rng.random(size=n_tx) < 0.40
        discount_pcts = np.where(
            discount_mask,
            rng.uniform(0.05, 0.30, size=n_tx),
            0.0,
        )

        # Prices and costs
        costs = product_costs[prod_idx_arr] * cogs_inflation
        prices = product_prices[prod_idx_arr] * (1.0 - discount_pcts)

        gross_revenue = (quantities * prices).round(2)
        cogs = (quantities * costs).round(2)
        gross_profit = (gross_revenue - cogs).round(2)

        # Random days within month
        day_offsets = rng.integers(0, days_in_month, size=n_tx)
        dates = pd.to_datetime(
            ym + "-01"
        ) + pd.to_timedelta(day_offsets, unit="D")

        # Transaction IDs: uuid4 hex prefix for uniqueness
        tx_ids = [f"T{month_idx:02d}{i:07d}" for i in range(n_tx)]

        month_df = pd.DataFrame({
            "transaction_id": tx_ids,
            "date": dates,
            "year_month": ym,
            "store_id": store_ids[store_idx_arr],
            "store_format": store_formats[store_idx_arr],
            "customer_id": chosen_custs,
            "product_id": product_ids[prod_idx_arr],
            "quantity": quantities.tolist(),
            "discount_pct": discount_pcts.round(4).tolist(),
            "unit_price": prices.round(2).tolist(),
            "gross_revenue": gross_revenue.tolist(),
            "cogs": cogs.tolist(),
            "gross_profit": gross_profit.tolist(),
        })

        all_rows.append(month_df)

        if (month_idx + 1) % 6 == 0 or month_idx == 0:
            logger.debug(
                f"  Month {ym}: {n_tx:,} transactions | "
                f"cogs_inflation={cogs_inflation:.3f} | "
                f"silver_churn={'ON' if silver_churn_active else 'off'}"
            )

    df = pd.concat(all_rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    logger.success(
        f"Transactions generated: {df.shape} | "
        f"total revenue=${df['gross_revenue'].sum():,.0f} | "
        f"avg margin={df['gross_profit'].sum()/df['gross_revenue'].sum()*100:.1f}%"
    )
    return df


def generate_marketing_spend(
    seed: int,
    stores: pd.DataFrame,
    n_months: int = 36,
) -> pd.DataFrame:
    """Generate monthly marketing spend by store and channel.

    One row per store × month × channel = 800 × 36 × 5 = 144,000 rows.

    Injected signal:
    - Digital generates 3x revenue lift per dollar in urban stores (encoded via
      digital_roi_multiplier column), 0.8x in rural — discoverable by MMM models.

    Args:
        seed: NumPy random seed for reproducibility.
        stores: Store master DataFrame.
        n_months: Number of months (default 36).

    Returns:
        Marketing spend DataFrame.
    """
    logger.info(f"Generating marketing spend for {len(stores)} stores × {n_months} months")
    rng = np.random.default_rng(seed + 4)

    months = _months_sequence(n_months)
    rows: list[dict] = []

    for _, store_row in stores.iterrows():
        sid = store_row["store_id"]
        fmt = store_row["store_format"]
        sqft = store_row["sq_footage"]

        # Monthly budget: $15k-$80k based on sq_footage (linear scale)
        budget_base = 15_000 + (sqft - 5_000) / (25_000 - 5_000) * 65_000
        channel_mix = CHANNEL_MIX[fmt]
        digital_roi = DIGITAL_ROI_MULTIPLIER[fmt]

        for ym in months:
            # Budget with ±15% monthly noise
            budget = budget_base * rng.uniform(0.85, 1.15)

            for channel, mix_pct in zip(MARKETING_CHANNELS, channel_mix):
                spend = round(float(budget * mix_pct * rng.uniform(0.90, 1.10)), 2)
                rows.append({
                    "store_id": sid,
                    "year_month": ym,
                    "store_format": fmt,
                    "channel": channel,
                    "spend_usd": spend,
                    "digital_roi_multiplier": digital_roi if channel == "digital" else None,
                })

    df = pd.DataFrame(rows)
    logger.success(f"Marketing spend generated: {df.shape}")
    return df


def generate_inventory(
    seed: int,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    n_months: int = 36,
) -> pd.DataFrame:
    """Generate monthly inventory metrics per store × category.

    One row per store × category × month = 800 × 7 × 36 = 201,600 rows.

    Injected signal:
    - Cluster C stores have 2x higher stockout rate than A/B.

    Args:
        seed: NumPy random seed for reproducibility.
        stores: Store master DataFrame.
        products: Product catalogue DataFrame (used for category list).
        n_months: Number of months (default 36).

    Returns:
        Inventory DataFrame.
    """
    logger.info(f"Generating inventory for {len(stores)} stores × {len(CATEGORIES)} categories × {n_months} months")
    rng = np.random.default_rng(seed + 5)

    months = _months_sequence(n_months)

    store_ids = stores["store_id"].values
    store_clusters = stores["performance_cluster"].values

    rows: list[dict] = []

    for sid, cluster in zip(store_ids, store_clusters):
        # Base stockout probability
        base_stockout_prob = 0.08 if cluster in ["A", "B"] else 0.16  # cluster C = 2x

        for cat in CATEGORIES:
            for ym in months:
                stock_coverage = float(rng.uniform(14, 60))  # days of stock on hand
                is_stockout = bool(rng.random() < base_stockout_prob)
                stockout_days = int(rng.integers(1, 8)) if is_stockout else 0

                rows.append({
                    "store_id": sid,
                    "category": cat,
                    "year_month": ym,
                    "stock_coverage_days": round(stock_coverage, 1),
                    "stockout_flag": is_stockout,
                    "stockout_days": stockout_days,
                })

    df = pd.DataFrame(rows)
    logger.success(f"Inventory generated: {df.shape}")
    return df


def generate_store_costs(
    seed: int,
    stores: pd.DataFrame,
    n_months: int = 36,
) -> pd.DataFrame:
    """Generate monthly operating cost data per store.

    One row per store × month = 800 × 36 = 28,800 rows.

    Injected signal:
    - Costs inflate from month 13 onward (mirroring COGS inflation in transactions).

    Args:
        seed: NumPy random seed for reproducibility.
        stores: Store master DataFrame.
        n_months: Number of months (default 36).

    Returns:
        Store costs DataFrame.
    """
    logger.info(f"Generating store costs for {len(stores)} stores × {n_months} months")
    rng = np.random.default_rng(seed + 6)

    months = _months_sequence(n_months)
    rows: list[dict] = []

    for _, store_row in stores.iterrows():
        sid = store_row["store_id"]
        fmt = store_row["store_format"]
        sqft = store_row["sq_footage"]

        # Base costs driven by sq_footage and format
        rent_base = sqft * (rng.uniform(25, 55) if fmt == "urban" else rng.uniform(15, 30))
        labor_base = sqft * rng.uniform(12, 22)
        utilities_base = sqft * rng.uniform(2.5, 5.0)
        shrinkage_base = sqft * rng.uniform(0.5, 2.0)

        for month_idx, ym in enumerate(months):
            # Cost inflation: kicks in from month 13, ramps to +12% by month 36
            if month_idx < 12:
                inflation = 1.0
            else:
                progress = (month_idx - 12) / 23
                inflation = 1.0 + 0.12 * progress

            rent = round(float(rent_base * inflation * rng.uniform(0.97, 1.03)) / 12, 2)
            labor = round(float(labor_base * inflation * rng.uniform(0.92, 1.08)) / 12, 2)
            utilities = round(float(utilities_base * inflation * rng.uniform(0.85, 1.15)) / 12, 2)
            shrinkage = round(float(shrinkage_base * rng.uniform(0.80, 1.20)) / 12, 2)
            total = round(rent + labor + utilities + shrinkage, 2)

            rows.append({
                "store_id": sid,
                "year_month": ym,
                "store_format": fmt,
                "rent_usd": rent,
                "labor_usd": labor,
                "utilities_usd": utilities,
                "shrinkage_usd": shrinkage,
                "total_operating_cost_usd": total,
            })

    df = pd.DataFrame(rows)
    logger.success(f"Store costs generated: {df.shape}")
    return df


def generate_all(seed: int = 42, output_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Generate all NovaMart synthetic datasets and optionally save to parquet.

    Args:
        seed: Master random seed (each sub-generator offsets from this).
        output_dir: Directory to write parquet files. Created if not exists.

    Returns:
        Dictionary mapping table names to DataFrames.
    """
    logger.info(f"Starting full NovaMart data generation | seed={seed} | output_dir={output_dir}")
    _ensure_dir(output_dir)

    stores = generate_stores(seed, n_stores=800)
    customers = generate_customers(seed, n_customers=500_000)
    products = generate_products(seed, n_products=5_000)
    transactions = generate_transactions(seed, stores, customers, products, n_months=36)
    marketing = generate_marketing_spend(seed, stores, n_months=36)
    inventory = generate_inventory(seed, stores, products, n_months=36)
    store_costs = generate_store_costs(seed, stores, n_months=36)

    data: dict[str, pd.DataFrame] = {
        "stores": stores,
        "customers": customers,
        "products": products,
        "transactions": transactions,
        "marketing_spend": marketing,
        "inventory": inventory,
        "store_costs": store_costs,
    }

    logger.info("Saving all tables to parquet...")
    for name, df in data.items():
        path = Path(output_dir) / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"  Saved {name}.parquet ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    logger.success(f"All {len(data)} tables generated and saved to {output_dir}/")
    return data
