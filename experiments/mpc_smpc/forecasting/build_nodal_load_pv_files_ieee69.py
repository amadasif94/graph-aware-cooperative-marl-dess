from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PROJECT_ROOT / "data" / "time_series" / "processed_15min_smartds_ieee69.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "mpc_smpc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_YEAR_OUT = OUT_DIR / "ieee69_15min_nodal_load_pv_full_year.csv"
TRAIN_OUT = OUT_DIR / "ieee69_15min_nodal_load_pv_train_jan_aug.csv"

NUM_BUSES = 69

TRAIN_START = "2018-01-01 00:00:00"
TRAIN_END = "2018-08-31 23:59:59"


def find_column(df, prefix_candidates, node_idx):
    for prefix in prefix_candidates:
        col = f"{prefix}_{node_idx}"
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find node {node_idx} column for prefixes: {prefix_candidates}"
    )


def main():
    print("===================================================")
    print("BUILD IEEE69 NODAL LOAD/PV FILES")
    print("===================================================")
    print(f"Input: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if "date_time" not in df.columns:
        raise ValueError("Input file must contain a date_time column.")

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None)

    if df["date_time"].isna().any():
        raise ValueError("Some date_time values could not be parsed.")

    load_prefixes = [
        "active_power_node",
        "load_node",
        "load_kw_node",
    ]

    pv_prefixes = [
        "renewable_active_power_node",
        "pv_power_node",
        "pv_node",
        "solar_node",
    ]

    output_columns = {"date_time": df["date_time"]}

    if "price" in df.columns:
        output_columns["price"] = pd.to_numeric(
            df["price"],
            errors="coerce",
        ).fillna(0.0)

    load_cols = []
    pv_cols = []

    for bus in range(1, NUM_BUSES + 1):
        load_col = find_column(df, load_prefixes, bus)
        pv_col = find_column(df, pv_prefixes, bus)

        load_out_col = f"load_node_{bus}"
        pv_out_col = f"pv_node_{bus}"

        output_columns[load_out_col] = pd.to_numeric(
            df[load_col],
            errors="coerce",
        ).fillna(0.0)

        output_columns[pv_out_col] = pd.to_numeric(
            df[pv_col],
            errors="coerce",
        ).fillna(0.0)

        load_cols.append(load_out_col)
        pv_cols.append(pv_out_col)

    out = pd.DataFrame(output_columns)
    out["total_load_kw"] = out[load_cols].sum(axis=1)
    out["total_pv_kw"] = out[pv_cols].sum(axis=1)

    out = out.sort_values("date_time").reset_index(drop=True)
    out.to_csv(FULL_YEAR_OUT, index=False)

    train_mask = (
        (out["date_time"] >= pd.Timestamp(TRAIN_START))
        & (out["date_time"] <= pd.Timestamp(TRAIN_END))
    )

    train_df = out.loc[train_mask].copy()
    train_df.to_csv(TRAIN_OUT, index=False)

    print("Done.")
    print(f"Full-year rows: {len(out)}")
    print(f"Train Jan-Aug rows: {len(train_df)}")
    print(f"Full-year date range: {out['date_time'].min()} to {out['date_time'].max()}")
    print(
        f"Train date range:     {train_df['date_time'].min()} "
        f"to {train_df['date_time'].max()}"
    )
    print(f"Saved full-year file: {FULL_YEAR_OUT}")
    print(f"Saved train file:     {TRAIN_OUT}")


if __name__ == "__main__":
    main()