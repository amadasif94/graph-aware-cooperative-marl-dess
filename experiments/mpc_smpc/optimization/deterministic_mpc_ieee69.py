# ============================================================
# deterministic_mpc_ieee69.py
# ============================================================
# IEEE69 deterministic MPC using AutoGluon point forecasts.
#
# Expected point forecast file:
#   results/mpc_smpc/forecasts/forecast_69bus_load_pv_sep_dec_wide.csv
#
# Expected forecast columns:
#   forecast_load_node_1 ... forecast_load_node_69
#   forecast_pv_node_1   ... forecast_pv_node_69
#
# Solver reuse:
#   Uses stochastic_mpc_ieee69.py with S=1.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from experiments.mpc_smpc.optimization import baseline_config_ieee69 as cfg
from experiments.mpc_smpc.optimization.stochastic_mpc_ieee69 import (
    build_stochastic_mpc_problem,
    solve_stochastic_mpc,
)


@dataclass
class PointForecastData:
    timestamps: pd.DatetimeIndex
    forecast_load_kw: np.ndarray                 # [T, 69]
    forecast_pv_kw: np.ndarray                   # [T, 69]
    forecast_netload_kw: np.ndarray              # [T, 69]
    aggregate_load_forecast_kw: np.ndarray       # [T]
    aggregate_pv_forecast_kw: np.ndarray         # [T]
    aggregate_netload_forecast_kw: np.ndarray    # [T]
    price: Optional[np.ndarray] = None           # [T]
    raw_df: Optional[pd.DataFrame] = None


def _normalize_topology_case(topology_case: str = "TP1") -> str:
    if hasattr(cfg, "normalize_topology_case"):
        return cfg.normalize_topology_case(topology_case)
    return str(topology_case or "TP1").upper()


def _find_datetime_col(df: pd.DataFrame) -> str:
    candidates = ["date_time", "datetime", "datetime_utc", "timestamp", "time"]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(f"No datetime column found. Columns: {list(df.columns)}")


def _find_price_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "price",
        "lambda",
        "lbmp",
        "DAM Zonal LBMP",
        "DAM Zonal LBMP ($/MWh)",
    ]

    lower_map = {c.lower().strip(): c for c in df.columns}

    for c in candidates:
        key = c.lower().strip()
        if key in lower_map:
            return lower_map[key]

    return None


def _load_cols(prefix: str) -> list[str]:
    return [f"{prefix}_node_{i}" for i in range(1, cfg.NUM_BUSES + 1)]


def _strict_15min_check(ts: pd.DatetimeIndex, label: str) -> None:
    if ts.has_duplicates:
        raise ValueError(f"{label}: duplicate timestamps found.")

    if not ts.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps are not sorted.")

    diffs = ts.to_series().diff().dropna()

    if len(diffs) > 0 and not (diffs == pd.Timedelta(minutes=15)).all():
        bad = diffs[diffs != pd.Timedelta(minutes=15)].head(10)
        raise ValueError(f"{label}: timestamps are not strictly 15-min.\n{bad}")


def load_point_forecast_data(
    forecast_file: Path = cfg.FORECAST_WIDE_FILE,
    max_steps: Optional[int] = cfg.MAX_STEPS,
) -> PointForecastData:
    forecast_file = Path(forecast_file)

    if not forecast_file.exists():
        raise FileNotFoundError(f"Point forecast file not found: {forecast_file}")

    df = pd.read_csv(forecast_file)

    if df.empty:
        raise ValueError(f"Point forecast file is empty: {forecast_file}")

    dt_col = _find_datetime_col(df)

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    if df[dt_col].isna().any():
        raise ValueError("Point forecast file contains invalid datetime values.")

    df = df.sort_values(dt_col).reset_index(drop=True)

    if "forecast_month" in df.columns:
        df = df[df["forecast_month"].astype(str) == "2018-12"].copy()
        df = df.sort_values(dt_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("No December rows found in point forecast file.")

    timestamps = pd.DatetimeIndex(pd.to_datetime(df[dt_col]))
    _strict_15min_check(timestamps, "point forecast timestamps")

    load_cols = _load_cols("forecast_load")
    pv_cols = _load_cols("forecast_pv")

    missing_load = [c for c in load_cols if c not in df.columns]
    missing_pv = [c for c in pv_cols if c not in df.columns]

    if missing_load:
        raise ValueError(f"Missing forecast load columns: {missing_load[:10]}")

    if missing_pv:
        raise ValueError(f"Missing forecast PV columns: {missing_pv[:10]}")

    forecast_load_kw = df[load_cols].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    forecast_pv_kw = df[pv_cols].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(forecast_load_kw).all():
        raise ValueError("forecast_load_kw contains NaN/inf.")

    if not np.isfinite(forecast_pv_kw).all():
        raise ValueError("forecast_pv_kw contains NaN/inf.")

    forecast_load_kw = np.maximum(forecast_load_kw, 0.0)
    forecast_pv_kw = np.maximum(forecast_pv_kw, 0.0)

    forecast_netload_kw = forecast_load_kw - forecast_pv_kw

    aggregate_load_forecast_kw = np.sum(forecast_load_kw, axis=1)
    aggregate_pv_forecast_kw = np.sum(forecast_pv_kw, axis=1)
    aggregate_netload_forecast_kw = np.sum(forecast_netload_kw, axis=1)

    price = None
    price_col = _find_price_col(df)

    if price_col is not None:
        price = pd.to_numeric(
            df[price_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.isfinite(price).all():
            raise ValueError(f"Price column {price_col} contains NaN/inf.")

    if max_steps is not None:
        max_steps = int(max_steps)

        if max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")

        timestamps = timestamps[:max_steps]
        forecast_load_kw = forecast_load_kw[:max_steps, :]
        forecast_pv_kw = forecast_pv_kw[:max_steps, :]
        forecast_netload_kw = forecast_netload_kw[:max_steps, :]

        aggregate_load_forecast_kw = aggregate_load_forecast_kw[:max_steps]
        aggregate_pv_forecast_kw = aggregate_pv_forecast_kw[:max_steps]
        aggregate_netload_forecast_kw = aggregate_netload_forecast_kw[:max_steps]

        if price is not None:
            price = price[:max_steps]

    return PointForecastData(
        timestamps=timestamps,
        forecast_load_kw=forecast_load_kw,
        forecast_pv_kw=forecast_pv_kw,
        forecast_netload_kw=forecast_netload_kw,
        aggregate_load_forecast_kw=aggregate_load_forecast_kw,
        aggregate_pv_forecast_kw=aggregate_pv_forecast_kw,
        aggregate_netload_forecast_kw=aggregate_netload_forecast_kw,
        price=price,
        raw_df=df,
    )


def get_point_forecast_window(
    data: PointForecastData,
    t0: int,
    horizon_t: int,
) -> Dict[str, np.ndarray]:
    t0 = int(t0)
    horizon_t = int(horizon_t)

    if t0 < 0:
        raise ValueError("t0 must be nonnegative.")

    if horizon_t <= 0:
        raise ValueError("horizon_t must be positive.")

    t1 = t0 + horizon_t

    if t1 > len(data.timestamps):
        raise ValueError(
            f"Window [{t0}:{t1}] exceeds available T={len(data.timestamps)}"
        )

    return {
        "timestamps": data.timestamps[t0:t1],
        "aggregate_netload_kw": data.aggregate_netload_forecast_kw[t0:t1].copy(),
        "load_kw": data.forecast_load_kw[t0:t1, :].copy(),
        "pv_kw": data.forecast_pv_kw[t0:t1, :].copy(),
        "netload_kw": data.forecast_netload_kw[t0:t1, :].copy(),
    }


def get_price_window_from_point_forecast(
    data: PointForecastData,
    t0: int,
    horizon_t: int,
    fallback_price: Optional[np.ndarray] = None,
) -> np.ndarray:
    t0 = int(t0)
    horizon_t = int(horizon_t)
    t1 = t0 + horizon_t

    if data.price is not None:
        if t1 > len(data.price):
            raise ValueError("Price window exceeds point forecast price length.")
        return data.price[t0:t1].copy()

    if fallback_price is not None:
        fallback_price = np.asarray(fallback_price, dtype=np.float64).reshape(-1)
        if t1 > len(fallback_price):
            raise ValueError("Price window exceeds fallback price length.")
        return fallback_price[t0:t1].copy()

    print("Warning: no price found. Using zero price for deterministic MPC.")
    return np.zeros(horizon_t, dtype=np.float64)


def solve_deterministic_mpc(
    aggregate_netload_T_kw: np.ndarray,
    price_T: np.ndarray,
    soc0: np.ndarray,
    load_T69_kw: Optional[np.ndarray] = None,
    pv_T69_kw: Optional[np.ndarray] = None,
    load_T33_kw: Optional[np.ndarray] = None,
    pv_T33_kw: Optional[np.ndarray] = None,
    topology_config: Optional[dict] = None,
    topology_case: str = "TP1",
    enforce_network_first_action: bool = True,
    solver_name: str = cfg.SOLVER_NAME,
    verbose: bool = cfg.VERBOSE,
    threads: int = cfg.THREADS,
    mip_gap: float = cfg.MIP_GAP,
    time_limit: float = cfg.TIME_LIMIT,
    warm_start: bool = cfg.WARM_START,
) -> Dict[str, Any]:

    topology_case = _normalize_topology_case(topology_case)

    if load_T69_kw is None and load_T33_kw is not None:
        load_T69_kw = load_T33_kw

    if pv_T69_kw is None and pv_T33_kw is not None:
        pv_T69_kw = pv_T33_kw

    aggregate_netload_T_kw = np.asarray(
        aggregate_netload_T_kw,
        dtype=np.float64,
    ).reshape(-1)

    price_T = np.asarray(price_T, dtype=np.float64).reshape(-1)
    soc0 = np.asarray(soc0, dtype=np.float64).reshape(-1)

    T = len(aggregate_netload_T_kw)

    if T <= 0:
        raise ValueError("aggregate_netload_T_kw is empty.")

    if len(price_T) != T:
        raise ValueError(f"price_T length {len(price_T)} != T={T}")

    if len(soc0) != cfg.NUM_DESS:
        raise ValueError(f"soc0 length {len(soc0)} != NUM_DESS={cfg.NUM_DESS}")

    if not np.isfinite(aggregate_netload_T_kw).all():
        raise ValueError("aggregate_netload_T_kw contains NaN/inf.")

    if not np.isfinite(price_T).all():
        raise ValueError("price_T contains NaN/inf.")

    if not np.isfinite(soc0).all():
        raise ValueError("soc0 contains NaN/inf.")

    load_ST69_kw = None
    pv_ST69_kw = None

    if load_T69_kw is not None and pv_T69_kw is not None:
        load_T69_kw = np.asarray(load_T69_kw, dtype=np.float64)
        pv_T69_kw = np.asarray(pv_T69_kw, dtype=np.float64)

        expected_shape = (T, cfg.NUM_BUSES)

        if load_T69_kw.shape != expected_shape:
            raise ValueError(
                f"load_T69_kw shape {load_T69_kw.shape} != {expected_shape}"
            )

        if pv_T69_kw.shape != expected_shape:
            raise ValueError(
                f"pv_T69_kw shape {pv_T69_kw.shape} != {expected_shape}"
            )

        if not np.isfinite(load_T69_kw).all():
            raise ValueError("load_T69_kw contains NaN/inf.")

        if not np.isfinite(pv_T69_kw).all():
            raise ValueError("pv_T69_kw contains NaN/inf.")

        load_ST69_kw = load_T69_kw.reshape(1, T, cfg.NUM_BUSES)
        pv_ST69_kw = pv_T69_kw.reshape(1, T, cfg.NUM_BUSES)

    bundle = build_stochastic_mpc_problem(
        S=1,
        T=T,
        num_dess=cfg.NUM_DESS,
        dt_hours=cfg.DT_HOURS,
        soc_min=cfg.SOC_MIN,
        soc_max=cfg.SOC_MAX,
        eta_ch=cfg.ETA_CH,
        eta_dis=cfg.ETA_DIS,
        p_ch_max_kw=cfg.P_CH_MAX_KW,
        p_dis_max_kw=cfg.P_DIS_MAX_KW,
        grid_export_limit_kw=cfg.GRID_EXPORT_LIMIT_KW,
        u_unmet_max_kw=cfg.U_UNMET_MAX_KW,
        u_curt_max_kw=cfg.U_CURT_MAX_KW,
    )

    sol = solve_stochastic_mpc(
        bundle=bundle,
        aggregate_netload_ST_kw=aggregate_netload_T_kw.reshape(1, T),
        price_T=price_T,
        soc0=soc0,
        load_ST69_kw=load_ST69_kw,
        pv_ST69_kw=pv_ST69_kw,
        enforce_network_first_action=enforce_network_first_action,
        topology_config=topology_config,
        topology_case=topology_case,
        solver_name=solver_name,
        verbose=verbose,
        threads=threads,
        mip_gap=mip_gap,
        time_limit=time_limit,
        warm_start=warm_start,
    )

    sol["controller"] = "deterministic_mpc"
    sol["forecast_source"] = str(cfg.FORECAST_WIDE_FILE)
    sol["topology_case"] = topology_case

    return sol


def solve_first_action(
    point_data: PointForecastData,
    t0: int,
    horizon_t: int,
    soc0: np.ndarray,
    fallback_price: Optional[np.ndarray] = None,
    topology_config: Optional[dict] = None,
    topology_case: str = "TP1",
    enforce_network_first_action: bool = True,
) -> Dict[str, Any]:
    topology_case = _normalize_topology_case(topology_case)

    win = get_point_forecast_window(
        data=point_data,
        t0=t0,
        horizon_t=horizon_t,
    )

    price_T = get_price_window_from_point_forecast(
        data=point_data,
        t0=t0,
        horizon_t=horizon_t,
        fallback_price=fallback_price,
    )

    return solve_deterministic_mpc(
        aggregate_netload_T_kw=win["aggregate_netload_kw"],
        price_T=price_T,
        soc0=soc0,
        load_T69_kw=win["load_kw"],
        pv_T69_kw=win["pv_kw"],
        topology_config=topology_config,
        topology_case=topology_case,
        enforce_network_first_action=enforce_network_first_action,
    )


def point_forecast_summary(data: PointForecastData) -> pd.DataFrame:
    T = len(data.timestamps)

    return pd.DataFrame([{
        "system": "ieee69",
        "T": int(T),
        "num_buses": int(data.forecast_load_kw.shape[1]),
        "start_time": str(data.timestamps[0]),
        "end_time": str(data.timestamps[-1]),
        "forecast_load_min_kw": float(np.min(data.forecast_load_kw)),
        "forecast_load_max_kw": float(np.max(data.forecast_load_kw)),
        "forecast_pv_min_kw": float(np.min(data.forecast_pv_kw)),
        "forecast_pv_max_kw": float(np.max(data.forecast_pv_kw)),
        "aggregate_netload_min_kw": float(np.min(data.aggregate_netload_forecast_kw)),
        "aggregate_netload_max_kw": float(np.max(data.aggregate_netload_forecast_kw)),
        "aggregate_netload_mean_kw": float(np.mean(data.aggregate_netload_forecast_kw)),
        "has_price": bool(data.price is not None),
    }])


if __name__ == "__main__":
    data = load_point_forecast_data()
    print(point_forecast_summary(data).to_string(index=False))