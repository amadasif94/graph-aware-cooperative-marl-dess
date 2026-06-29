# ============================================================
# scenario_data.py
# ============================================================
# Load and validate IEEE33 SARIMA load/PV scenario tensors.
#
# Current SARIMA files contain:
#   scenarios_kw shape = [T_DEC, 33 buses, N_SCEN]
#
# This module converts them into forms needed by MPC/SMPC:
#   load_scenarios_kw[t, bus, s]
#   pv_scenarios_kw[t, bus, s]
#   netload_scenarios_kw[t, bus, s] = load - pv
#   aggregate_netload_scenarios_kw[t, s] = sum_bus(load - pv)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from experiments.mpc_smpc.optimization import baseline_config as cfg


@dataclass
class ScenarioData:
    timestamps: pd.DatetimeIndex
    bus_numbers: np.ndarray

    load_scenarios_kw: np.ndarray          # [T, 33, S]
    pv_scenarios_kw: np.ndarray            # [T, 33, S]
    netload_scenarios_kw: np.ndarray       # [T, 33, S]

    aggregate_load_scenarios_kw: np.ndarray       # [T, S]
    aggregate_pv_scenarios_kw: np.ndarray         # [T, S]
    aggregate_netload_scenarios_kw: np.ndarray    # [T, S]

    price: Optional[np.ndarray] = None             # [T]
    forecast_df: Optional[pd.DataFrame] = None


def _load_npz(path: Path, label: str) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{label} scenario file not found: {path}")

    data = np.load(path, allow_pickle=True)

    required = ["timestamps", "bus_numbers", "scenarios_kw"]

    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(
            f"{label} scenario file missing arrays: {missing}. "
            f"Available arrays: {data.files}"
        )

    return {
        "timestamps": data["timestamps"],
        "bus_numbers": data["bus_numbers"],
        "scenarios_kw": data["scenarios_kw"],
        "residual_scenarios_kw": (
            data["residual_scenarios_kw"]
            if "residual_scenarios_kw" in data.files
            else None
        ),
    }


def _parse_timestamps(raw_timestamps, label: str) -> pd.DatetimeIndex:
    ts = pd.to_datetime(raw_timestamps, errors="coerce")

    if pd.isna(ts).any():
        raise ValueError(f"{label}: invalid timestamps found.")

    ts = pd.DatetimeIndex(ts)

    if ts.has_duplicates:
        raise ValueError(f"{label}: duplicate timestamps found.")

    if not ts.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps are not sorted.")

    diffs = ts.to_series().diff().dropna()

    if len(diffs) > 0 and not (diffs == pd.Timedelta(minutes=15)).all():
        bad = diffs[diffs != pd.Timedelta(minutes=15)].head(10)
        raise ValueError(f"{label}: timestamps are not strictly 15-min.\n{bad}")

    return ts


def _validate_tensor(x: np.ndarray, label: str, expected_buses: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError(f"{label}: expected 3D tensor [T, buses, scenarios], got {x.shape}")

    if x.shape[1] != expected_buses:
        raise ValueError(
            f"{label}: expected {expected_buses} buses, got shape {x.shape}"
        )

    if not np.isfinite(x).all():
        raise ValueError(f"{label}: tensor contains NaN or inf.")

    if (x < -1e-9).any():
        raise ValueError(f"{label}: tensor contains negative values.")

    x = np.maximum(x, 0.0)

    return x


def _load_price_from_forecast_file(
    forecast_path: Path,
    timestamps: pd.DatetimeIndex,
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    forecast_path = Path(forecast_path)

    if not forecast_path.exists():
        print(f"Warning: forecast file not found, price will be None: {forecast_path}")
        return None, None

    df = pd.read_csv(forecast_path)

    dt_candidates = ["date_time", "datetime", "datetime_utc", "timestamp", "time"]
    dt_col = None

    for c in dt_candidates:
        if c in df.columns:
            dt_col = c
            break

    if dt_col is None:
        print("Warning: could not find datetime column in forecast file. price=None")
        return None, df

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    if df[dt_col].isna().any():
        raise ValueError("Forecast file contains invalid datetime values.")

    df = df.sort_values(dt_col).drop_duplicates(subset=[dt_col]).set_index(dt_col)

    price_col = None
    for c in ["price", "lambda", "lbmp", "DAM Zonal LBMP"]:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        print("Warning: could not find price column in forecast file. price=None")
        return None, df.reset_index()

    aligned = df.reindex(timestamps)

    if aligned[price_col].isna().any():
        missing = int(aligned[price_col].isna().sum())
        raise ValueError(f"Price alignment failed. Missing price rows: {missing}")

    price = pd.to_numeric(aligned[price_col], errors="coerce").to_numpy(dtype=np.float64)

    if not np.isfinite(price).all():
        raise ValueError("Aligned price contains NaN or inf.")

    return price, df.reset_index()


def load_scenario_data(
    load_scenario_file: Path = cfg.LOAD_SCENARIO_FILE,
    pv_scenario_file: Path = cfg.PV_SCENARIO_FILE,
    forecast_file: Path = cfg.FORECAST_WIDE_FILE,
    max_steps: Optional[int] = cfg.MAX_STEPS,
    max_scenarios: Optional[int] = None,
) -> ScenarioData:
    """
    Load load/PV SARIMA scenarios and build net-load scenario tensors.

    Returns
    -------
    ScenarioData
        Object containing nodal and aggregate scenario arrays.
    """

    load_npz = _load_npz(Path(load_scenario_file), label="load")
    pv_npz = _load_npz(Path(pv_scenario_file), label="pv")

    load_ts = _parse_timestamps(load_npz["timestamps"], label="load")
    pv_ts = _parse_timestamps(pv_npz["timestamps"], label="pv")

    if len(load_ts) != len(pv_ts) or not np.all(load_ts == pv_ts):
        raise ValueError("Load and PV timestamps do not match.")

    load_bus_numbers = np.asarray(load_npz["bus_numbers"], dtype=int)
    pv_bus_numbers = np.asarray(pv_npz["bus_numbers"], dtype=int)

    if not np.array_equal(load_bus_numbers, pv_bus_numbers):
        raise ValueError("Load and PV bus number arrays do not match.")

    expected_bus_numbers = np.arange(1, cfg.NUM_BUSES + 1, dtype=int)

    if not np.array_equal(load_bus_numbers, expected_bus_numbers):
        raise ValueError(
            f"Unexpected bus numbers. Expected {expected_bus_numbers}, got {load_bus_numbers}"
        )

    load = _validate_tensor(
        load_npz["scenarios_kw"],
        label="load_scenarios_kw",
        expected_buses=cfg.NUM_BUSES,
    )

    pv = _validate_tensor(
        pv_npz["scenarios_kw"],
        label="pv_scenarios_kw",
        expected_buses=cfg.NUM_BUSES,
    )

    if load.shape != pv.shape:
        raise ValueError(f"Load/PV scenario shape mismatch: {load.shape} vs {pv.shape}")

    timestamps = load_ts

    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")

        load = load[:max_steps, :, :]
        pv = pv[:max_steps, :, :]
        timestamps = timestamps[:max_steps]

    if max_scenarios is not None:
        max_scenarios = int(max_scenarios)
        if max_scenarios <= 0:
            raise ValueError("max_scenarios must be positive or None.")
        if max_scenarios > load.shape[2]:
            raise ValueError(
                f"Requested max_scenarios={max_scenarios}, "
                f"but only {load.shape[2]} are available."
            )

        load = load[:, :, :max_scenarios]
        pv = pv[:, :, :max_scenarios]

    netload = load - pv

    aggregate_load = np.sum(load, axis=1)
    aggregate_pv = np.sum(pv, axis=1)
    aggregate_netload = np.sum(netload, axis=1)

    price, forecast_df = _load_price_from_forecast_file(
        forecast_path=Path(forecast_file),
        timestamps=timestamps,
    )

    out = ScenarioData(
        timestamps=timestamps,
        bus_numbers=load_bus_numbers,
        load_scenarios_kw=load,
        pv_scenarios_kw=pv,
        netload_scenarios_kw=netload,
        aggregate_load_scenarios_kw=aggregate_load,
        aggregate_pv_scenarios_kw=aggregate_pv,
        aggregate_netload_scenarios_kw=aggregate_netload,
        price=price,
        forecast_df=forecast_df,
    )

    validate_scenario_data(out)

    return out


def validate_scenario_data(data: ScenarioData) -> None:
    T, B, S = data.load_scenarios_kw.shape

    if data.pv_scenarios_kw.shape != (T, B, S):
        raise ValueError("PV scenario shape mismatch.")

    if data.netload_scenarios_kw.shape != (T, B, S):
        raise ValueError("Net-load scenario shape mismatch.")

    if B != cfg.NUM_BUSES:
        raise ValueError(f"Expected {cfg.NUM_BUSES} buses, got {B}.")

    if len(data.timestamps) != T:
        raise ValueError("Timestamp length mismatch.")

    if data.aggregate_load_scenarios_kw.shape != (T, S):
        raise ValueError("Aggregate load shape mismatch.")

    if data.aggregate_pv_scenarios_kw.shape != (T, S):
        raise ValueError("Aggregate PV shape mismatch.")

    if data.aggregate_netload_scenarios_kw.shape != (T, S):
        raise ValueError("Aggregate net-load shape mismatch.")

    if data.price is not None and len(data.price) != T:
        raise ValueError("Price length mismatch.")

    arrays = [
        ("load_scenarios_kw", data.load_scenarios_kw),
        ("pv_scenarios_kw", data.pv_scenarios_kw),
        ("netload_scenarios_kw", data.netload_scenarios_kw),
        ("aggregate_load_scenarios_kw", data.aggregate_load_scenarios_kw),
        ("aggregate_pv_scenarios_kw", data.aggregate_pv_scenarios_kw),
        ("aggregate_netload_scenarios_kw", data.aggregate_netload_scenarios_kw),
    ]

    for name, arr in arrays:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or inf.")


def get_aggregate_smpc_window(
    data: ScenarioData,
    t0: int,
    horizon_t: int,
    n_scenarios: int,
) -> np.ndarray:
    """
    Return aggregate net-load SMPC window.

    Output shape:
        [S, T]

    This matches your old SMPC solver convention:
        netload_ST = [scenario, time]
    """
    t0 = int(t0)
    horizon_t = int(horizon_t)
    n_scenarios = int(n_scenarios)

    if t0 < 0:
        raise ValueError("t0 must be nonnegative.")

    if horizon_t <= 0:
        raise ValueError("horizon_t must be positive.")

    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")

    t1 = t0 + horizon_t

    if t1 > data.aggregate_netload_scenarios_kw.shape[0]:
        raise ValueError(
            f"Window [{t0}:{t1}] exceeds available T={data.aggregate_netload_scenarios_kw.shape[0]}"
        )

    if n_scenarios > data.aggregate_netload_scenarios_kw.shape[1]:
        raise ValueError(
            f"Requested {n_scenarios} scenarios, but only "
            f"{data.aggregate_netload_scenarios_kw.shape[1]} are available."
        )

    # [T, S] -> [S, T]
    return data.aggregate_netload_scenarios_kw[t0:t1, :n_scenarios].T.copy()


def get_nodal_smpc_window(
    data: ScenarioData,
    t0: int,
    horizon_t: int,
    n_scenarios: int,
) -> Dict[str, np.ndarray]:
    """
    Return nodal load/PV/net-load SMPC window.

    Output shapes:
        load_kw    [S, T, 33]
        pv_kw      [S, T, 33]
        netload_kw [S, T, 33]
    """
    t0 = int(t0)
    horizon_t = int(horizon_t)
    n_scenarios = int(n_scenarios)

    t1 = t0 + horizon_t

    if t0 < 0 or t1 > data.load_scenarios_kw.shape[0]:
        raise ValueError("Invalid time window.")

    if n_scenarios <= 0 or n_scenarios > data.load_scenarios_kw.shape[2]:
        raise ValueError("Invalid n_scenarios.")

    return {
        "load_kw": np.transpose(data.load_scenarios_kw[t0:t1, :, :n_scenarios], (2, 0, 1)).copy(),
        "pv_kw": np.transpose(data.pv_scenarios_kw[t0:t1, :, :n_scenarios], (2, 0, 1)).copy(),
        "netload_kw": np.transpose(data.netload_scenarios_kw[t0:t1, :, :n_scenarios], (2, 0, 1)).copy(),
    }


def get_price_window(data: ScenarioData, t0: int, horizon_t: int) -> Optional[np.ndarray]:
    if data.price is None:
        return None

    t0 = int(t0)
    t1 = t0 + int(horizon_t)

    if t0 < 0 or t1 > len(data.price):
        raise ValueError("Invalid price window.")

    return data.price[t0:t1].copy()


def scenario_summary(data: ScenarioData) -> pd.DataFrame:
    T, B, S = data.load_scenarios_kw.shape

    return pd.DataFrame([{
        "T": int(T),
        "num_buses": int(B),
        "num_scenarios": int(S),
        "start_time": str(data.timestamps[0]),
        "end_time": str(data.timestamps[-1]),

        "load_min_kw": float(np.min(data.load_scenarios_kw)),
        "load_max_kw": float(np.max(data.load_scenarios_kw)),
        "pv_min_kw": float(np.min(data.pv_scenarios_kw)),
        "pv_max_kw": float(np.max(data.pv_scenarios_kw)),

        "aggregate_netload_min_kw": float(np.min(data.aggregate_netload_scenarios_kw)),
        "aggregate_netload_max_kw": float(np.max(data.aggregate_netload_scenarios_kw)),
        "aggregate_netload_mean_kw": float(np.mean(data.aggregate_netload_scenarios_kw)),
    }])


if __name__ == "__main__":
    data = load_scenario_data()
    print(scenario_summary(data).to_string(index=False))