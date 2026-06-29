"""
Build 15-minute IEEE69 node-level time-series data from SMART-DS load/PV
and NYISO hourly LBMP price.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "smartds_raw"
LOAD_DIR = RAW_DIR / "Load_Data"
SOLAR_DIR = RAW_DIR / "Solar_Data"
PRICE_PATH = RAW_DIR / "nyiso_2018_lbmp.csv"

OUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "time_series"
    / "processed_15min_smartds_ieee69.csv"
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# IEEE69 settings
# ============================================================

NUM_BUSES = 69

# Use more SMART-DS profiles for larger feeder
NUM_LOAD_FILES = 69
NUM_SOLAR_FILES = 69

LOAD_FILES_PER_BUS = NUM_LOAD_FILES // NUM_BUSES
SOLAR_FILES_PER_BUS = NUM_SOLAR_FILES // NUM_BUSES


# ============================================================
# Time settings
# ============================================================

N_15MIN_STEPS = 365 * 24 * 4
START_UTC = "2018-01-01 00:00:00+00:00"


# ============================================================
# Scaling settings
# ============================================================

# IEEE69 is substantially larger than IEEE33.
# Start conservatively to avoid severe undervoltage
# during first feasibility testing.

LOAD_SCALING_FACTOR = 0.68

# Each SMART-DS solar CSV is a 1000 kW reference system
PV_SCALING_FACTOR = 0.05


def make_master_index():
    return pd.date_range(
        start=START_UTC,
        periods=N_15MIN_STEPS,
        freq="15min",
    )


def find_price_column(df):

    candidates = [
        "DAM Zonal LBMP",
        "DAM Zonal LBMP ($/MWh)",
        "LBMP",
        "price",
        "Price",
    ]

    lower_map = {c.lower().strip(): c for c in df.columns}

    for cand in candidates:

        key = cand.lower().strip()

        if key in lower_map:
            return lower_map[key]

    raise ValueError(
        f"Could not identify price column. "
        f"Columns: {list(df.columns)}"
    )


def read_load_file(path):

    df = pd.read_parquet(path)

    if "total_site_electricity_kw" not in df.columns:
        raise ValueError(
            f"{path.name} missing total_site_electricity_kw column."
        )

    if len(df) != N_15MIN_STEPS:
        raise ValueError(
            f"{path.name} has {len(df)} rows, "
            f"expected {N_15MIN_STEPS}."
        )

    load_kw = pd.to_numeric(
        df["total_site_electricity_kw"],
        errors="coerce",
    ).fillna(0.0).reset_index(drop=True)

    return LOAD_SCALING_FACTOR * load_kw


def read_solar_file(path):

    df = pd.read_csv(path)

    pv_col = "kW Generated (1000 kW Array)"

    if pv_col not in df.columns:
        raise ValueError(
            f"{path.name} missing PV column: {pv_col}"
        )

    if len(df) != N_15MIN_STEPS:
        raise ValueError(
            f"{path.name} has {len(df)} rows, "
            f"expected {N_15MIN_STEPS}."
        )

    pv_kw = pd.to_numeric(
        df[pv_col],
        errors="coerce",
    ).fillna(0.0).reset_index(drop=True)

    return PV_SCALING_FACTOR * pv_kw


def read_price_file(path, master_index):

    df = pd.read_csv(path)

    price_col = find_price_column(df)

    price = pd.to_numeric(
        df[price_col],
        errors="coerce",
    ).ffill().bfill().reset_index(drop=True)

    # hourly -> 15-minute
    price_15min = price.repeat(4).reset_index(drop=True)

    if len(price_15min) < N_15MIN_STEPS:
        raise ValueError(
            f"Price file gives only {len(price_15min)} "
            f"15-min values, expected at least {N_15MIN_STEPS}."
        )

    price_15min = (
        price_15min.iloc[:N_15MIN_STEPS]
        .reset_index(drop=True)
    )

    out = pd.DataFrame()

    out["date_time"] = master_index
    out["price"] = price_15min

    return out


def aggregate_profiles_to_buses(
    files,
    reader_func,
    num_files,
    files_per_bus,
    prefix,
    master_index,
):

    files = sorted(files)[:num_files]

    if len(files) < num_files:
        raise ValueError(
            f"Need {num_files} files for {prefix}, "
            f"found {len(files)}."
        )

    out = pd.DataFrame()

    out["date_time"] = master_index

    for bus_idx in range(NUM_BUSES):

        selected = files[
            bus_idx * files_per_bus:
            (bus_idx + 1) * files_per_bus
        ]

        total = pd.Series(
            0.0,
            index=range(N_15MIN_STEPS),
        )

        for path in selected:
            total += reader_func(path)

        out[f"{prefix}_node_{bus_idx + 1}"] = total.values

    return out


def main():

    print("===================================================")
    print("SMART-DS IEEE69 15-MIN PREPROCESSING")
    print("===================================================")

    if NUM_LOAD_FILES % NUM_BUSES != 0:
        raise ValueError(
            "NUM_LOAD_FILES must be divisible by NUM_BUSES."
        )

    if NUM_SOLAR_FILES % NUM_BUSES != 0:
        raise ValueError(
            "NUM_SOLAR_FILES must be divisible by NUM_BUSES."
        )

    load_files = sorted(LOAD_DIR.glob("*.parquet"))
    solar_files = sorted(SOLAR_DIR.glob("*.csv"))

    print(f"Found load parquet files: {len(load_files)}")
    print(f"Found solar CSV files:    {len(solar_files)}")

    print(f"Using load files:         {NUM_LOAD_FILES}")
    print(f"Using solar files:        {NUM_SOLAR_FILES}")

    print(f"Load files per bus:       {LOAD_FILES_PER_BUS}")
    print(f"Solar files per bus:      {SOLAR_FILES_PER_BUS}")

    print(f"Load scaling factor:      {LOAD_SCALING_FACTOR}")
    print(f"PV scaling factor:        {PV_SCALING_FACTOR}")

    print(f"Expected rows:            {N_15MIN_STEPS}")

    master_index = make_master_index()

    print("\nAggregating load data...")

    load_bus = aggregate_profiles_to_buses(
        files=load_files,
        reader_func=read_load_file,
        num_files=NUM_LOAD_FILES,
        files_per_bus=LOAD_FILES_PER_BUS,
        prefix="active_power",
        master_index=master_index,
    )

    print("Aggregating solar data...")

    solar_bus = aggregate_profiles_to_buses(
        files=solar_files,
        reader_func=read_solar_file,
        num_files=NUM_SOLAR_FILES,
        files_per_bus=SOLAR_FILES_PER_BUS,
        prefix="renewable_active_power",
        master_index=master_index,
    )

    print("Processing price data...")

    price = read_price_file(
        PRICE_PATH,
        master_index,
    )

    print("Merging final dataset...")

    df = load_bus.merge(
        solar_bus,
        on="date_time",
        how="inner",
    )

    df = df.merge(
        price,
        on="date_time",
        how="inner",
    )

    if len(df) != N_15MIN_STEPS:
        raise ValueError(
            f"Final dataframe has {len(df)} rows, "
            f"expected {N_15MIN_STEPS}."
        )

    ordered_cols = ["date_time"]

    ordered_cols += [
        f"active_power_node_{i}"
        for i in range(1, NUM_BUSES + 1)
    ]

    ordered_cols += [
        f"renewable_active_power_node_{i}"
        for i in range(1, NUM_BUSES + 1)
    ]

    ordered_cols += ["price"]

    df = df[ordered_cols]

    df["date_time"] = (
        df["date_time"]
        .dt.strftime("%Y-%m-%d %H:%M:%S%z")
    )

    df["date_time"] = df["date_time"].str.replace(
        r"(\+0000)$",
        "+00:00",
        regex=True,
    )

    df.to_csv(OUT_PATH, index=False)

    # ============================================================
    # Statistics
    # ============================================================

    load_cols = [
        f"active_power_node_{i}"
        for i in range(1, NUM_BUSES + 1)
    ]

    pv_cols = [
        f"renewable_active_power_node_{i}"
        for i in range(1, NUM_BUSES + 1)
    ]

    total_load_kw = df[load_cols].sum(axis=1)
    total_pv_kw = df[pv_cols].sum(axis=1)

    print("\n===================================================")
    print("DONE")
    print("===================================================")

    print(f"Output file: {OUT_PATH}")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print(f"Start: {df['date_time'].iloc[0]}")
    print(f"End:   {df['date_time'].iloc[-1]}")

    print("\nFeeder load statistics:")
    print(f"  mean kW: {total_load_kw.mean():.3f}")
    print(f"  max  kW: {total_load_kw.max():.3f}")
    print(f"  min  kW: {total_load_kw.min():.3f}")

    print("\nFeeder PV statistics:")
    print(f"  mean kW: {total_pv_kw.mean():.3f}")
    print(f"  max  kW: {total_pv_kw.max():.3f}")
    print(f"  min  kW: {total_pv_kw.min():.3f}")

    print("\nPrice statistics:")
    print(f"  mean: {df['price'].mean():.3f}")
    print(f"  max : {df['price'].max():.3f}")
    print(f"  min : {df['price'].min():.3f}")


if __name__ == "__main__":
    main()