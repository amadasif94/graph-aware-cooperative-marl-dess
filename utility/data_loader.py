"""
Data loading utilities for the IEEE 33-bus graph-aware cooperative MARL
framework for distributed energy storage system (DESS) coordination.

This module loads and validates:

1. Static IEEE 33-bus network data:
    - bus_data.csv
    - line_data.csv
    - dess_buses.csv

2. Processed 15-minute SMART-DS/NYISO node-level time-series data:
    - processed_15min_smartds.csv

Expected processed time-series format:

    date_time,
    active_power_node_1, ..., active_power_node_N,
    renewable_active_power_node_1, ..., renewable_active_power_node_N,
    price

All bus indices in static network files use zero-based indexing.
The processed time-series columns use one-based node labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Required static-data columns
# ============================================================

REQUIRED_BUS_COLUMNS = ["bus", "load_kw", "load_kvar", "pv_kw"]
REQUIRED_LINE_COLUMNS = ["from_bus", "to_bus", "r_ohm", "x_ohm"]
REQUIRED_DESS_COLUMNS = ["bus"]


# ============================================================
# Generic validation utilities
# ============================================================

def _check_file_exists(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError("File does not exist: {}".format(path))

    return path


def _check_not_empty(path):
    path = Path(path)

    if path.stat().st_size == 0:
        raise ValueError("File is empty: {}".format(path))


def _check_columns(df, required_columns, file_name):
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "{} is missing required columns: {}".format(file_name, missing)
        )


def _coerce_numeric_columns(df, columns, file_name):
    df = df.copy()

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[columns].isna().any().any():
        bad_cols = df[columns].columns[df[columns].isna().any()].tolist()
        raise ValueError(
            "{} contains invalid numeric values in columns: {}".format(
                file_name,
                bad_cols,
            )
        )

    return df


# ============================================================
# Static network data loaders
# ============================================================

def load_bus_data(path):
    """
    Load static bus-level network data.

    Expected columns:
        bus, load_kw, load_kvar, pv_kw
    """

    path = _check_file_exists(path)
    _check_not_empty(path)

    bus_df = pd.read_csv(path)
    _check_columns(bus_df, REQUIRED_BUS_COLUMNS, str(path))

    bus_df = _coerce_numeric_columns(
        bus_df,
        REQUIRED_BUS_COLUMNS,
        str(path),
    )

    bus_df["bus"] = bus_df["bus"].astype(int)
    bus_df["load_kw"] = bus_df["load_kw"].astype(float)
    bus_df["load_kvar"] = bus_df["load_kvar"].astype(float)
    bus_df["pv_kw"] = bus_df["pv_kw"].astype(float)

    return bus_df.sort_values("bus").reset_index(drop=True)


def load_line_data(path):
    """
    Load static line-level network data.

    Required columns:
        from_bus, to_bus, r_ohm, x_ohm

    Optional columns:
        status,
        max_i_ka,
        max_i_a,
        max_current_a,
        current_limit_pu
    """

    path = _check_file_exists(path)
    _check_not_empty(path)

    line_df = pd.read_csv(path)
    _check_columns(line_df, REQUIRED_LINE_COLUMNS, str(path))

    numeric_cols = list(REQUIRED_LINE_COLUMNS)

    optional_numeric_cols = [
        "status",
        "max_i_ka",
        "max_i_a",
        "max_current_a",
        "current_limit_pu",
    ]

    for col in optional_numeric_cols:
        if col in line_df.columns:
            numeric_cols.append(col)

    line_df = _coerce_numeric_columns(line_df, numeric_cols, str(path))

    line_df["from_bus"] = line_df["from_bus"].astype(int)
    line_df["to_bus"] = line_df["to_bus"].astype(int)
    line_df["r_ohm"] = line_df["r_ohm"].astype(float)
    line_df["x_ohm"] = line_df["x_ohm"].astype(float)

    if "status" not in line_df.columns:
        line_df["status"] = 1.0

    line_df["status"] = line_df["status"].astype(float)

    return line_df.reset_index(drop=True)


def load_dess_buses(path):
    """
    Load DESS placement data.

    Expected column:
        bus

    Returns zero-based DESS bus indices.
    """

    path = _check_file_exists(path)
    _check_not_empty(path)

    dess_df = pd.read_csv(path)
    _check_columns(dess_df, REQUIRED_DESS_COLUMNS, str(path))

    dess_df = _coerce_numeric_columns(
        dess_df,
        REQUIRED_DESS_COLUMNS,
        str(path),
    )

    return dess_df["bus"].astype(int).tolist()


# ============================================================
# Static network validation
# ============================================================

def validate_static_network_data(bus_df, line_df, dess_buses, config):
    """
    Validate static IEEE feeder data against the configuration dictionary.
    """

    num_buses = int(config["network"]["num_buses"])
    slack_bus = int(config["network"]["slack_bus"])

    # --------------------------------------------------------
    # Bus checks
    # --------------------------------------------------------
    if len(bus_df) != num_buses:
        raise ValueError(
            "Expected {} buses, found {}.".format(num_buses, len(bus_df))
        )

    expected_buses = set(range(num_buses))
    actual_buses = set(bus_df["bus"].astype(int).tolist())

    if actual_buses != expected_buses:
        raise ValueError(
            "Bus indices must be exactly 0 to {}. Found: {}".format(
                num_buses - 1,
                sorted(actual_buses),
            )
        )

    if slack_bus < 0 or slack_bus >= num_buses:
        raise ValueError("Slack bus is outside the valid bus range.")

    if (bus_df["load_kw"] < 0.0).any():
        raise ValueError("Static bus load values must be nonnegative.")

    if (bus_df["load_kvar"] < 0.0).any():
        raise ValueError("Static bus reactive load values must be nonnegative.")

    if (bus_df["pv_kw"] < 0.0).any():
        raise ValueError("Static bus PV values must be nonnegative.")

    # --------------------------------------------------------
    # Line checks
    # --------------------------------------------------------
    if len(line_df) == 0:
        raise ValueError("Line dataframe cannot be empty.")

    for line_idx, row in line_df.iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])

        if from_bus < 0 or from_bus >= num_buses:
            raise ValueError(
                "Invalid from_bus in line {}: {}".format(line_idx, from_bus)
            )

        if to_bus < 0 or to_bus >= num_buses:
            raise ValueError(
                "Invalid to_bus in line {}: {}".format(line_idx, to_bus)
            )

        if from_bus == to_bus:
            raise ValueError(
                "Line {} has identical from_bus and to_bus.".format(line_idx)
            )

        if float(row["r_ohm"]) < 0.0:
            raise ValueError(
                "Line resistance cannot be negative in line {}.".format(line_idx)
            )

        if float(row["x_ohm"]) < 0.0:
            raise ValueError(
                "Line reactance cannot be negative in line {}.".format(line_idx)
            )

        if "status" in line_df.columns:
            if float(row["status"]) not in [0.0, 1.0]:
                raise ValueError(
                    "Line status must be 0 or 1 in line {}.".format(line_idx)
                )

    # --------------------------------------------------------
    # DESS checks
    # --------------------------------------------------------
    expected_num_dess = int(config["dess"]["num_dess"])

    if len(dess_buses) != expected_num_dess:
        raise ValueError(
            "Expected {} DESS units, found {}.".format(
                expected_num_dess,
                len(dess_buses),
            )
        )

    if len(set(dess_buses)) != len(dess_buses):
        raise ValueError("DESS bus list contains duplicate entries.")

    for bus in dess_buses:
        if bus < 0 or bus >= num_buses:
            raise ValueError(
                "DESS bus {} is outside valid range [0, {}].".format(
                    bus,
                    num_buses - 1,
                )
            )

        if bus == slack_bus:
            raise ValueError("DESS should not be placed at the slack bus.")

    return True


def load_static_network_data(config):
    """
    Load all static network data.
    """

    paths = config["paths"]

    bus_df = load_bus_data(paths["bus_data"])
    line_df = load_line_data(paths["line_data"])
    dess_buses = load_dess_buses(paths["dess_buses"])

    validate_static_network_data(
        bus_df=bus_df,
        line_df=line_df,
        dess_buses=dess_buses,
        config=config,
    )

    return {
        "bus_data": bus_df,
        "line_data": line_df,
        "dess_buses": dess_buses,
    }


def load_ieee33_static_data(config):
    """
    Backward-compatible wrapper for IEEE 33-bus static data.
    """

    return load_static_network_data(config)


# ============================================================
# Processed node-level time-series loader
# ============================================================

def load_processed_node_time_series(path, num_buses, allow_empty=True):
    """
    Load processed node-level time-series data.

    Expected columns:
        date_time,
        active_power_node_1, ..., active_power_node_N,
        renewable_active_power_node_1, ..., renewable_active_power_node_N,
        price
    """

    path = _check_file_exists(path)

    if path.stat().st_size == 0:
        if allow_empty:
            return None
        raise ValueError("Processed time-series file is empty: {}".format(path))

    df = pd.read_csv(path)

    required_base_cols = ["date_time", "price"]
    _check_columns(df, required_base_cols, str(path))

    load_cols = [
        "active_power_node_{}".format(i)
        for i in range(1, num_buses + 1)
    ]

    renewable_cols = [
        "renewable_active_power_node_{}".format(i)
        for i in range(1, num_buses + 1)
    ]

    _check_columns(df, load_cols, str(path))
    _check_columns(df, renewable_cols, str(path))

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        utc=True,
        errors="coerce",
    )

    if df["date_time"].isna().any():
        bad_count = int(df["date_time"].isna().sum())
        raise ValueError(
            "Processed time-series file contains {} invalid date_time values.".format(
                bad_count
            )
        )

    numeric_cols = load_cols + renewable_cols + ["price"]
    df = _coerce_numeric_columns(df, numeric_cols, str(path))

    if (df[load_cols] < 0.0).any().any():
        raise ValueError("Time-series load values must be nonnegative.")

    if (df[renewable_cols] < 0.0).any().any():
        raise ValueError("Time-series renewable generation values must be nonnegative.")

    df = df.sort_values("date_time").reset_index(drop=True)

    if df["date_time"].duplicated().any():
        dup_count = int(df["date_time"].duplicated().sum())
        raise ValueError(
            "Processed time-series file contains {} duplicate timestamps.".format(
                dup_count
            )
        )

    return {
        "df": df,
        "load_cols": load_cols,
        "renewable_cols": renewable_cols,
        "price_col": "price",
        "datetime_col": "date_time",
    }


def validate_time_series_resolution(processed_time_series, config):
    """
    Validate that the processed time-series file matches the configured
    simulation resolution.

    For the current SMART-DS setup, the expected resolution is 15 minutes.
    """

    if processed_time_series is None:
        return True

    df = processed_time_series["df"]
    datetime_col = processed_time_series["datetime_col"]

    if len(df) < 2:
        raise ValueError("Processed time-series file must contain at least two rows.")

    expected_minutes = int(config["time"]["time_step_minutes"])
    expected_delta = pd.Timedelta(minutes=expected_minutes)

    deltas = df[datetime_col].diff().dropna()

    bad = deltas[deltas != expected_delta]

    if len(bad) > 0:
        raise ValueError(
            "Processed time-series resolution mismatch. Expected {}, but found "
            "{} irregular intervals.".format(expected_delta, len(bad))
        )

    return True


def select_time_series_row(processed_time_series, index):
    """
    Select one row from processed node-level time-series data.

    Returns:
        date_time : pandas.Timestamp
        load_kw   : np.ndarray, shape (num_buses,)
        pv_kw     : np.ndarray, shape (num_buses,)
        price     : float
    """

    df = processed_time_series["df"]
    load_cols = processed_time_series["load_cols"]
    renewable_cols = processed_time_series["renewable_cols"]
    price_col = processed_time_series["price_col"]
    datetime_col = processed_time_series["datetime_col"]

    index = int(index)

    if index < 0 or index >= len(df):
        raise IndexError(
            "Time-series index {} outside valid range [0, {}].".format(
                index,
                len(df) - 1,
            )
        )

    row = df.iloc[index]

    return {
        "date_time": row[datetime_col],
        "load_kw": row[load_cols].to_numpy(dtype=np.float64),
        "pv_kw": row[renewable_cols].to_numpy(dtype=np.float64),
        "price": float(row[price_col]),
    }


# ============================================================
# Train / validation / test split utilities
# ============================================================

def split_time_series_by_date(processed_time_series, config):
    """
    Split processed time-series data into train, validation, and test indices.
    """

    if processed_time_series is None:
        return {
            "train_indices": [],
            "val_indices": [],
            "test_indices": [],
        }

    if "splits" not in config:
        raise ValueError("Config is missing the 'splits' section.")

    df = processed_time_series["df"]
    datetime_col = processed_time_series["datetime_col"]
    splits = config["splits"]

    def _indices_between(start_date, end_date):
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)

        mask = (df[datetime_col] >= start) & (df[datetime_col] < end)
        return df.index[mask].tolist()

    train_indices = _indices_between(
        splits["train_start"],
        splits["train_end"],
    )

    val_indices = _indices_between(
        splits["val_start"],
        splits["val_end"],
    )

    test_indices = _indices_between(
        splits["test_start"],
        splits["test_end"],
    )

    if len(train_indices) == 0:
        raise ValueError("Training split contains no samples.")

    if len(val_indices) == 0:
        raise ValueError("Validation split contains no samples.")

    if len(test_indices) == 0:
        raise ValueError("Test split contains no samples.")

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


# ============================================================
# Full data loader
# ============================================================

def load_all_data(config, require_time_series=False):
    """
    Load static network data, processed time-series data, and split indices.
    """

    static_data = load_static_network_data(config)
    paths = config["paths"]

    processed_ts = load_processed_node_time_series(
        path=paths["processed_time_series"],
        num_buses=int(config["network"]["num_buses"]),
        allow_empty=not require_time_series,
    )

    if processed_ts is None and require_time_series:
        raise ValueError("Processed time-series data are required but unavailable.")

    validate_time_series_resolution(processed_ts, config)

    split_indices = split_time_series_by_date(processed_ts, config)

    return {
        "static": static_data,
        "processed_time_series": processed_ts,
        "splits": split_indices,
    }


def load_ieee33_all_data(config, require_time_series=False):
    """
    Backward-compatible wrapper for IEEE 33-bus data loading.
    """

    return load_all_data(
        config=config,
        require_time_series=require_time_series,
    )