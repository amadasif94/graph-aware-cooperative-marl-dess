#!/usr/bin/env python3
# ============================================================
# 69-BUS LOAD/PV SARIMA RESIDUAL SCENARIO GENERATION
# Robust version:
#   - Keeps IEEE69 paths/output names the same.
#   - Uses representative-bus grid search for LOAD and PV orders.
#   - Fits every bus using selected orders.
#   - If final SARIMA fit does not converge, uses residual-bootstrap fallback.
#   - If statsmodels simulate() fails, including "SVD did not converge",
#     uses residual-bootstrap fallback for that scenario path.
# ============================================================

import os
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ============================================================
# HPC THREAD SAFETY
# ============================================================

os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "1")
os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ.get("VECLIB_MAXIMUM_THREADS", "1")

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(".").resolve()

FORECAST_DIR = PROJECT_ROOT / "results" / "mpc_smpc" / "forecasts"
OUT_DIR = PROJECT_ROOT / "results" / "mpc_smpc" / "scenarios_ieee69"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDE_FORECAST_CSV = FORECAST_DIR / "forecast_69bus_load_pv_sep_dec_wide.csv"
RESID_ARCHIVE_CSV = FORECAST_DIR / "residual_archive_sep_nov_69bus_load_pv.csv"

OUT_LOAD_ORDER_CSV = OUT_DIR / "sarima_selected_order_load_69bus.csv"
OUT_PV_ORDER_CSV = OUT_DIR / "sarima_selected_order_pv_69bus.csv"

OUT_LOAD_GRID_CSV = OUT_DIR / "sarima_grid_search_load_representative_bus_69bus.csv"
OUT_PV_GRID_CSV = OUT_DIR / "sarima_grid_search_pv_representative_bus_69bus.csv"

OUT_LOAD_SCENARIOS_NPZ = OUT_DIR / "load_scenarios_december_69bus.npz"
OUT_PV_SCENARIOS_NPZ = OUT_DIR / "pv_scenarios_december_69bus.npz"

OUT_SUMMARY_JSON = OUT_DIR / "sarima_scenario_manifest_69bus.json"

# ============================================================
# SETTINGS
# ============================================================

NUM_BUSES = 69
N_SCEN = int(os.environ.get("SARIMA_N_SCEN", "100"))
SEASONAL_PERIOD = 96
RANDOM_SEED = 42

N_WORKERS = int(os.environ.get("SARIMA_N_WORKERS", "24"))
SARIMA_MAXITER = int(os.environ.get("SARIMA_MAXITER", "50"))

# Grid-search selection still prefers converged candidates.
SELECT_CONVERGED_ONLY = True

SKIP_GRID_IF_ORDER_EXISTS = True
FORCE_GRID_SEARCH = False
FORCE_REGENERATE_SCENARIOS = True

SARIMA_TREND = "n"

# Safer defaults for simulation stability. You may override from shell:
#   export SARIMA_ENFORCE_STATIONARITY=0
#   export SARIMA_ENFORCE_INVERTIBILITY=0
ENFORCE_STATIONARITY = bool(int(os.environ.get("SARIMA_ENFORCE_STATIONARITY", "1")))
ENFORCE_INVERTIBILITY = bool(int(os.environ.get("SARIMA_ENFORCE_INVERTIBILITY", "1")))

SPREAD_SCALE_LOAD = float(os.environ.get("SARIMA_SPREAD_SCALE_LOAD", "1.0"))
SPREAD_SCALE_PV = float(os.environ.get("SARIMA_SPREAD_SCALE_PV", "1.0"))

USE_RESIDUAL_CLIP = bool(int(os.environ.get("SARIMA_USE_RESIDUAL_CLIP", "0")))
RESID_CLIP_SIGMA = float(os.environ.get("SARIMA_RESID_CLIP_SIGMA", "4.0"))

# Fallback behavior:
# If final SARIMA fit does not converge or simulate() fails, use bootstrap.
ALLOW_NONCONVERGED_FINAL_FIT = False
BOOTSTRAP_BLOCK_SIZE = int(os.environ.get("SARIMA_BOOTSTRAP_BLOCK_SIZE", "96"))

P_LIST = [0, 1, 2, 3]
D_LIST = [0, 1]
Q_LIST = [0, 1, 2, 3]

SP_LIST = [0]
SD_LIST = [0, 1]
SQ_LIST = [0, 1]


# ============================================================
# COLUMN HELPERS
# ============================================================

def node_cols(prefix):
    return [f"{prefix}_node_{i}" for i in range(1, NUM_BUSES + 1)]


LOAD_RESID_COLS = node_cols("residual_load")
PV_RESID_COLS = node_cols("residual_pv")

LOAD_FORECAST_COLS = node_cols("forecast_load")
PV_FORECAST_COLS = node_cols("forecast_pv")


# ============================================================
# BASIC UTILITIES
# ============================================================

def find_datetime_col(df):
    candidates = ["date_time", "datetime", "datetime_utc", "timestamp", "time"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No datetime column found. Columns: {list(df.columns)}")


def ensure_required_cols(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns. "
            f"First missing columns: {missing[:10]} | total={len(missing)}"
        )


def strict_15min_check(ts, label):
    ts = pd.to_datetime(pd.Series(ts), errors="coerce")

    if ts.isna().any():
        raise ValueError(f"{label}: invalid datetime values found.")

    if ts.duplicated().any():
        raise ValueError(f"{label}: duplicate timestamps found.")

    if not ts.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps are not sorted.")

    diffs = ts.diff().dropna()
    if not (diffs == pd.Timedelta(minutes=15)).all():
        bad = diffs[diffs != pd.Timedelta(minutes=15)].head(10)
        raise ValueError(f"{label}: timestamps are not strictly 15-min.\n{bad}")

    return True


def quick_series_stats(x, name):
    x = pd.Series(x).dropna().astype(float)

    if len(x) == 0:
        return {
            "name": name,
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "median": np.nan,
            "q95": np.nan,
            "max": np.nan,
        }

    return {
        "name": name,
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q05": float(x.quantile(0.05)),
        "median": float(x.median()),
        "q95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def select_highest_variance_bus(df, residual_cols, label):
    variances = df[residual_cols].astype(float).var(axis=0, ddof=1)
    selected_col = variances.idxmax()
    selected_bus = int(selected_col.split("_node_")[-1])

    table = (
        variances.rename("variance")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("variance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nHighest-variance {label} residual bus:")
    print(f"  column = {selected_col}")
    print(f"  bus    = {selected_bus}")
    print(f"  var    = {float(variances[selected_col]):.6f}")

    return selected_col, selected_bus, table


def parse_order_row(row):
    return (
        (int(row["p"]), int(row["d"]), int(row["q"])),
        (int(row["P"]), int(row["D"]), int(row["Q"]), int(row["s"])),
    )


def load_selected_order(order_csv, label):
    if not order_csv.exists():
        raise FileNotFoundError(f"{label} selected-order CSV not found: {order_csv}")

    df = pd.read_csv(order_csv)
    if df.empty:
        raise ValueError(f"{label} selected-order CSV is empty: {order_csv}")

    row = df.iloc[0].to_dict()

    required = ["p", "d", "q", "P", "D", "Q", "s"]
    missing = [c for c in required if c not in row]
    if missing:
        raise ValueError(f"{label} selected-order CSV missing fields: {missing}")

    order, seasonal_order = parse_order_row(row)

    print(f"\nReloaded saved {label} SARIMA order:")
    print(f"  order          = {order}")
    print(f"  seasonal_order = {seasonal_order}")
    print(f"  source         = {order_csv}")

    return row


def clean_residual_array(ref_residuals, label):
    ref = pd.Series(ref_residuals).dropna().astype(float).to_numpy(dtype=float)

    if len(ref) == 0:
        raise ValueError(f"{label}: no valid residuals available for fallback.")

    ref = ref[np.isfinite(ref)]

    if len(ref) == 0:
        raise ValueError(f"{label}: residuals are all non-finite.")

    return ref


def residual_clip_bounds(ref_residuals):
    ref = pd.Series(ref_residuals).dropna().astype(float)
    mu = float(ref.mean())
    sigma = float(ref.std(ddof=1))

    if not np.isfinite(sigma) or sigma <= 0:
        return None, None

    lo = mu - RESID_CLIP_SIGMA * sigma
    hi = mu + RESID_CLIP_SIGMA * sigma

    return lo, hi


# ============================================================
# SARIMA FITTING / SEARCH
# ============================================================

def fit_one_candidate(args):
    (
        y_values,
        order,
        seasonal_order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        maxiter,
    ) = args

    p, d, q = order
    P, D, Q, s = seasonal_order

    out = {
        "p": p,
        "d": d,
        "q": q,
        "P": P,
        "D": D,
        "Q": Q,
        "s": s,
        "order": str(order),
        "seasonal_order": str(seasonal_order),
        "trend": trend,
        "enforce_stationarity": bool(enforce_stationarity),
        "enforce_invertibility": bool(enforce_invertibility),
        "converged": False,
        "aic": np.nan,
        "bic": np.nan,
        "hqic": np.nan,
        "llf": np.nan,
        "status": "failed",
    }

    try:
        model = SARIMAX(
            y_values,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        result = model.fit(disp=False, maxiter=maxiter)

        out.update({
            "converged": bool(getattr(result, "mle_retvals", {}).get("converged", True)),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "hqic": float(result.hqic),
            "llf": float(result.llf),
            "status": "ok",
        })

    except Exception as e:
        out["status"] = f"failed: {type(e).__name__}: {str(e)}"

    return out


def search_best_sarima(y, label):
    y = pd.Series(y).dropna().astype(float)

    if len(y) < 200:
        raise ValueError(f"{label}: too few samples for SARIMA search: {len(y)}")

    y_values = y.to_numpy(dtype=float)

    candidates = []
    for p in P_LIST:
        for d in D_LIST:
            for q in Q_LIST:
                for P in SP_LIST:
                    for D in SD_LIST:
                        for Q in SQ_LIST:
                            if (p, d, q) == (0, 0, 0) and (P, D, Q) == (0, 0, 0):
                                continue

                            candidates.append(
                                (
                                    y_values,
                                    (p, d, q),
                                    (P, D, Q, SEASONAL_PERIOD),
                                    SARIMA_TREND,
                                    ENFORCE_STATIONARITY,
                                    ENFORCE_INVERTIBILITY,
                                    SARIMA_MAXITER,
                                )
                            )

    print("\n" + "=" * 80)
    print(f"SARIMA SEARCH: {label}")
    print("=" * 80)
    print(f"Samples                : {len(y_values)}")
    print(f"Seasonal period        : {SEASONAL_PERIOD}")
    print(f"Candidates             : {len(candidates)}")
    print(f"Workers                : {N_WORKERS}")
    print(f"Maxiter                : {SARIMA_MAXITER}")
    print(f"Enforce stationarity   : {ENFORCE_STATIONARITY}")
    print(f"Enforce invertibility  : {ENFORCE_INVERTIBILITY}")

    rows = []
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(fit_one_candidate, c) for c in candidates]

        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({
                    "p": np.nan,
                    "d": np.nan,
                    "q": np.nan,
                    "P": np.nan,
                    "D": np.nan,
                    "Q": np.nan,
                    "s": SEASONAL_PERIOD,
                    "order": np.nan,
                    "seasonal_order": np.nan,
                    "trend": SARIMA_TREND,
                    "enforce_stationarity": ENFORCE_STATIONARITY,
                    "enforce_invertibility": ENFORCE_INVERTIBILITY,
                    "converged": False,
                    "aic": np.nan,
                    "bic": np.nan,
                    "hqic": np.nan,
                    "llf": np.nan,
                    "status": f"failed: {type(e).__name__}: {str(e)}",
                })

            done += 1

            if done % 25 == 0 or done == len(candidates):
                ok = sum(r["status"] == "ok" for r in rows)
                conv = sum((r["status"] == "ok") and bool(r["converged"]) for r in rows)
                print(f"Completed {done:4d}/{len(candidates)} | ok={ok:4d} | converged={conv:4d}")

    grid_df = pd.DataFrame(rows)

    ok_df = grid_df[grid_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise RuntimeError(f"{label}: all SARIMA candidates failed.")

    if SELECT_CONVERGED_ONLY:
        select_df = ok_df[ok_df["converged"] == True].copy()
        if select_df.empty:
            print(f"WARNING: {label}: no converged candidates. Falling back to finite-BIC non-converged candidates.")
            select_df = ok_df.copy()
    else:
        select_df = ok_df.copy()

    select_df = select_df[np.isfinite(select_df["bic"])].copy()
    if select_df.empty:
        raise RuntimeError(f"{label}: no finite-BIC candidates.")

    select_df = (
        select_df
        .sort_values(["bic", "aic", "hqic"], ascending=[True, True, True])
        .reset_index(drop=True)
    )

    best = select_df.iloc[0].to_dict()

    grid_df = (
        grid_df
        .sort_values(["status", "converged", "bic", "aic"], ascending=[True, False, True, True])
        .reset_index(drop=True)
    )

    print(f"\nSelected {label} SARIMA:")
    print(pd.DataFrame([best]))

    return best, grid_df


def fit_sarima(y, order, seasonal_order, label):
    """
    Fit final per-bus SARIMA.

    Returns
    -------
    result : statsmodels result or None
    fit_info : dict
        Contains converged/status/aic/bic/hqic/message.
    """

    y = pd.Series(y).dropna().astype(float)

    fit_info = {
        "label": label,
        "fit_status": "failed",
        "fit_message": "",
        "converged": False,
        "aic": np.nan,
        "bic": np.nan,
        "hqic": np.nan,
    }

    try:
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            trend=SARIMA_TREND,
            enforce_stationarity=ENFORCE_STATIONARITY,
            enforce_invertibility=ENFORCE_INVERTIBILITY,
        )

        result = model.fit(disp=False, maxiter=SARIMA_MAXITER)

        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))

        fit_info.update({
            "fit_status": "ok",
            "fit_message": "ok" if converged else "not_converged",
            "converged": converged,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "hqic": float(result.hqic),
        })

        if not converged:
            print(f"WARNING: final SARIMA fit did not converge for {label}. Bootstrap fallback will be used.")

        return result, fit_info

    except Exception as e:
        fit_info.update({
            "fit_status": "failed",
            "fit_message": f"{type(e).__name__}: {str(e)}",
            "converged": False,
        })
        print(f"WARNING: final SARIMA fit failed for {label}: {fit_info['fit_message']}. Bootstrap fallback will be used.")
        return None, fit_info


# ============================================================
# SCENARIO SIMULATION / FALLBACK
# ============================================================

def bootstrap_one_path(ref_residuals, horizon, rng, block_size=96):
    """
    Block bootstrap residual path.

    This is used as a robust fallback when SARIMA fitting/simulation is unstable.
    It preserves some short-run temporal structure better than iid sampling.
    """

    ref = clean_residual_array(ref_residuals, label="bootstrap")

    horizon = int(horizon)
    block_size = int(block_size)

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    if len(ref) == 1 or block_size <= 1:
        return rng.choice(ref, size=horizon, replace=True).astype(float)

    block_size = min(block_size, len(ref), horizon)

    chunks = []
    total = 0

    while total < horizon:
        max_start = len(ref) - block_size
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        block = ref[start:start + block_size]
        chunks.append(block)
        total += len(block)

    path = np.concatenate(chunks)[:horizon].astype(float)

    if len(path) != horizon:
        raise RuntimeError(f"Bootstrap length mismatch: {len(path)} != {horizon}")

    return path


def simulate_paths(
    fitted_result,
    horizon,
    n_scen,
    seed,
    ref_residuals,
    spread_scale=1.0,
    require_converged=True,
    label="series",
):
    """
    Simulate residual paths.

    Robust rule:
      - If fitted_result is None -> bootstrap all paths.
      - If final fit did not converge and require_converged=True -> bootstrap all paths.
      - If SARIMA simulate() fails for any scenario -> bootstrap that path.
    """

    rng = np.random.default_rng(seed)
    out = np.zeros((horizon, n_scen), dtype=np.float32)

    ref = clean_residual_array(ref_residuals, label=label)

    lo, hi = (None, None)
    if USE_RESIDUAL_CLIP:
        lo, hi = residual_clip_bounds(ref)
        if lo is None or hi is None:
            print(f"WARNING: {label}: invalid residual std for clipping; clipping disabled for this series.")

    fit_converged = False
    if fitted_result is not None:
        fit_converged = bool(getattr(fitted_result, "mle_retvals", {}).get("converged", True))

    use_sarima = (fitted_result is not None) and (fit_converged or not require_converged)

    stats = {
        "simulation_method": "sarima" if use_sarima else "bootstrap_all",
        "sarima_paths": 0,
        "bootstrap_paths": 0,
        "simulation_failures": 0,
        "first_simulation_error": "",
    }

    if not use_sarima:
        for s in range(n_scen):
            sim = bootstrap_one_path(
                ref_residuals=ref,
                horizon=horizon,
                rng=rng,
                block_size=BOOTSTRAP_BLOCK_SIZE,
            )

            if USE_RESIDUAL_CLIP and lo is not None and hi is not None:
                sim = np.clip(sim, lo, hi)

            out[:, s] = (spread_scale * sim).astype(np.float32)
            stats["bootstrap_paths"] += 1

        return out, stats

    for s in range(n_scen):
        seed_s = int(rng.integers(0, 2**32 - 1))

        try:
            sim = fitted_result.simulate(
                nsimulations=horizon,
                anchor="end",
                random_state=seed_s,
            )

            sim = np.asarray(sim, dtype=float).reshape(-1)

            if len(sim) != horizon:
                raise RuntimeError(f"Simulation length mismatch: {len(sim)} != {horizon}")

            if not np.isfinite(sim).all():
                raise RuntimeError("Non-finite simulated residual values.")

            stats["sarima_paths"] += 1

        except Exception as e:
            # This catches numpy.linalg.LinAlgError: SVD did not converge
            if stats["first_simulation_error"] == "":
                stats["first_simulation_error"] = f"{type(e).__name__}: {str(e)}"

            stats["simulation_failures"] += 1
            stats["bootstrap_paths"] += 1

            sim = bootstrap_one_path(
                ref_residuals=ref,
                horizon=horizon,
                rng=rng,
                block_size=BOOTSTRAP_BLOCK_SIZE,
            )

            if stats["simulation_failures"] == 1:
                print(
                    f"WARNING: {label}: SARIMA simulate() failed "
                    f"({stats['first_simulation_error']}). "
                    "Using bootstrap fallback for failed path(s)."
                )

        if USE_RESIDUAL_CLIP and lo is not None and hi is not None:
            sim = np.clip(sim, lo, hi)

        out[:, s] = (spread_scale * sim).astype(np.float32)

    if stats["bootstrap_paths"] > 0 and stats["sarima_paths"] > 0:
        stats["simulation_method"] = "sarima_with_bootstrap_fallback"
    elif stats["bootstrap_paths"] == n_scen:
        stats["simulation_method"] = "bootstrap_all"

    return out, stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("69-BUS LOAD/PV SARIMA SCENARIO GENERATION")
    print("=" * 80)

    print("\nRuntime settings:")
    print(f"  N_SCEN                       = {N_SCEN}")
    print(f"  N_WORKERS                    = {N_WORKERS}")
    print(f"  SARIMA_MAXITER               = {SARIMA_MAXITER}")
    print(f"  SEASONAL_PERIOD              = {SEASONAL_PERIOD}")
    print(f"  ENFORCE_STATIONARITY         = {ENFORCE_STATIONARITY}")
    print(f"  ENFORCE_INVERTIBILITY        = {ENFORCE_INVERTIBILITY}")
    print(f"  SKIP_GRID_IF_ORDER_EXISTS    = {SKIP_GRID_IF_ORDER_EXISTS}")
    print(f"  FORCE_GRID_SEARCH            = {FORCE_GRID_SEARCH}")
    print(f"  BOOTSTRAP_BLOCK_SIZE         = {BOOTSTRAP_BLOCK_SIZE}")
    print(f"  ALLOW_NONCONVERGED_FINAL_FIT = {ALLOW_NONCONVERGED_FINAL_FIT}")

    if not WIDE_FORECAST_CSV.exists():
        raise FileNotFoundError(WIDE_FORECAST_CSV)

    if not RESID_ARCHIVE_CSV.exists():
        raise FileNotFoundError(RESID_ARCHIVE_CSV)

    df_wide = pd.read_csv(WIDE_FORECAST_CSV)
    df_arch = pd.read_csv(RESID_ARCHIVE_CSV)

    dt_col_wide = find_datetime_col(df_wide)
    dt_col_arch = find_datetime_col(df_arch)

    df_wide[dt_col_wide] = pd.to_datetime(df_wide[dt_col_wide], errors="coerce")
    df_arch[dt_col_arch] = pd.to_datetime(df_arch[dt_col_arch], errors="coerce")

    df_wide = df_wide.sort_values(dt_col_wide).reset_index(drop=True)
    df_arch = df_arch.sort_values(dt_col_arch).reset_index(drop=True)

    ensure_required_cols(df_wide, LOAD_FORECAST_COLS + PV_FORECAST_COLS, "wide forecast")
    ensure_required_cols(df_arch, LOAD_RESID_COLS + PV_RESID_COLS, "residual archive")

    strict_15min_check(df_wide[dt_col_wide], "wide forecast")
    strict_15min_check(df_arch[dt_col_arch], "residual archive")

    if "forecast_month" in df_wide.columns:
        df_dec = df_wide[df_wide["forecast_month"].astype(str) == "2018-12"].copy()
    else:
        df_dec = df_wide[df_wide[dt_col_wide].dt.month == 12].copy()

    if df_dec.empty:
        raise ValueError("Could not find December rows in wide forecast file.")

    df_dec = df_dec.sort_values(dt_col_wide).reset_index(drop=True)

    dec_timestamps = pd.to_datetime(df_dec[dt_col_wide])
    strict_15min_check(dec_timestamps, "December forecast horizon")

    T_DEC = len(df_dec)

    print(f"\nResidual archive rows : {len(df_arch)}")
    print(f"December horizon rows : {T_DEC}")
    print(f"December start        : {dec_timestamps.min()}")
    print(f"December end          : {dec_timestamps.max()}")

    rep_load_col, rep_load_bus, load_var_table = select_highest_variance_bus(
        df_arch, LOAD_RESID_COLS, "load"
    )

    rep_pv_col, rep_pv_bus, pv_var_table = select_highest_variance_bus(
        df_arch, PV_RESID_COLS, "PV"
    )

    load_var_table.to_csv(OUT_DIR / "load_residual_variance_by_bus_69bus.csv", index=False)
    pv_var_table.to_csv(OUT_DIR / "pv_residual_variance_by_bus_69bus.csv", index=False)

    can_reload_load = (
        SKIP_GRID_IF_ORDER_EXISTS
        and OUT_LOAD_ORDER_CSV.exists()
        and not FORCE_GRID_SEARCH
    )
    can_reload_pv = (
        SKIP_GRID_IF_ORDER_EXISTS
        and OUT_PV_ORDER_CSV.exists()
        and not FORCE_GRID_SEARCH
    )

    if can_reload_load:
        best_load = load_selected_order(OUT_LOAD_ORDER_CSV, "load")
        load_grid_was_run = False
    else:
        best_load, grid_load = search_best_sarima(
            df_arch[rep_load_col].astype(float),
            label=f"load representative bus {rep_load_bus}",
        )
        grid_load.to_csv(OUT_LOAD_GRID_CSV, index=False)

        best_load_df = pd.DataFrame([
            {
                **best_load,
                "representative_bus": rep_load_bus,
                "representative_col": rep_load_col,
            }
        ])
        best_load_df.to_csv(OUT_LOAD_ORDER_CSV, index=False)
        load_grid_was_run = True

    if can_reload_pv:
        best_pv = load_selected_order(OUT_PV_ORDER_CSV, "PV")
        pv_grid_was_run = False
    else:
        best_pv, grid_pv = search_best_sarima(
            df_arch[rep_pv_col].astype(float),
            label=f"PV representative bus {rep_pv_bus}",
        )
        grid_pv.to_csv(OUT_PV_GRID_CSV, index=False)

        best_pv_df = pd.DataFrame([
            {
                **best_pv,
                "representative_bus": rep_pv_bus,
                "representative_col": rep_pv_col,
            }
        ])
        best_pv_df.to_csv(OUT_PV_ORDER_CSV, index=False)
        pv_grid_was_run = True

    load_order, load_seasonal_order = parse_order_row(best_load)
    pv_order, pv_seasonal_order = parse_order_row(best_pv)

    print("\nFinal selected orders:")
    print(f"  LOAD order          = {load_order}")
    print(f"  LOAD seasonal_order = {load_seasonal_order}")
    print(f"  PV order            = {pv_order}")
    print(f"  PV seasonal_order   = {pv_seasonal_order}")

    if (
        OUT_LOAD_SCENARIOS_NPZ.exists()
        and OUT_PV_SCENARIOS_NPZ.exists()
        and not FORCE_REGENERATE_SCENARIOS
    ):
        print("\nScenario NPZ files already exist and FORCE_REGENERATE_SCENARIOS=False.")
        print(f"Existing load scenarios: {OUT_LOAD_SCENARIOS_NPZ}")
        print(f"Existing PV scenarios  : {OUT_PV_SCENARIOS_NPZ}")
        return

    load_scenarios = np.zeros((T_DEC, NUM_BUSES, N_SCEN), dtype=np.float32)
    pv_scenarios = np.zeros((T_DEC, NUM_BUSES, N_SCEN), dtype=np.float32)

    load_residual_scenarios = np.zeros((T_DEC, NUM_BUSES, N_SCEN), dtype=np.float32)
    pv_residual_scenarios = np.zeros((T_DEC, NUM_BUSES, N_SCEN), dtype=np.float32)

    fit_rows = []

    print("\n" + "=" * 80)
    print("FITTING ALL 69 LOAD BUSES WITH SELECTED LOAD ORDER")
    print("=" * 80)

    for bus in range(1, NUM_BUSES + 1):
        resid_col = f"residual_load_node_{bus}"
        forecast_col = f"forecast_load_node_{bus}"

        y = df_arch[resid_col].astype(float)
        yhat_dec = df_dec[forecast_col].astype(float).to_numpy()

        print(f"LOAD bus {bus:02d}: fitting SARIMA")

        result, fit_info = fit_sarima(
            y=y,
            order=load_order,
            seasonal_order=load_seasonal_order,
            label=f"load_node_{bus}",
        )

        e_scen, sim_info = simulate_paths(
            fitted_result=result,
            horizon=T_DEC,
            n_scen=N_SCEN,
            seed=RANDOM_SEED + 1000 + bus,
            ref_residuals=y,
            spread_scale=SPREAD_SCALE_LOAD,
            require_converged=(not ALLOW_NONCONVERGED_FINAL_FIT),
            label=f"load_node_{bus}",
        )

        x_scen = yhat_dec[:, None] + e_scen
        x_scen = np.clip(x_scen, 0.0, None)

        load_residual_scenarios[:, bus - 1, :] = e_scen
        load_scenarios[:, bus - 1, :] = x_scen

        fit_rows.append({
            "kind": "load",
            "bus": bus,
            "residual_col": resid_col,
            "forecast_col": forecast_col,
            "order": str(load_order),
            "seasonal_order": str(load_seasonal_order),
            **fit_info,
            **sim_info,
            **quick_series_stats(e_scen.reshape(-1), f"load_residual_scenario_bus_{bus}"),
        })

    print("\n" + "=" * 80)
    print("FITTING ALL 69 PV BUSES WITH SELECTED PV ORDER")
    print("=" * 80)

    for bus in range(1, NUM_BUSES + 1):
        resid_col = f"residual_pv_node_{bus}"
        forecast_col = f"forecast_pv_node_{bus}"

        y = df_arch[resid_col].astype(float)
        yhat_dec = df_dec[forecast_col].astype(float).to_numpy()

        print(f"PV bus {bus:02d}: fitting SARIMA")

        result, fit_info = fit_sarima(
            y=y,
            order=pv_order,
            seasonal_order=pv_seasonal_order,
            label=f"pv_node_{bus}",
        )

        e_scen, sim_info = simulate_paths(
            fitted_result=result,
            horizon=T_DEC,
            n_scen=N_SCEN,
            seed=RANDOM_SEED + 2000 + bus,
            ref_residuals=y,
            spread_scale=SPREAD_SCALE_PV,
            require_converged=(not ALLOW_NONCONVERGED_FINAL_FIT),
            label=f"pv_node_{bus}",
        )

        x_scen = yhat_dec[:, None] + e_scen
        x_scen = np.clip(x_scen, 0.0, None)

        pv_residual_scenarios[:, bus - 1, :] = e_scen
        pv_scenarios[:, bus - 1, :] = x_scen

        fit_rows.append({
            "kind": "pv",
            "bus": bus,
            "residual_col": resid_col,
            "forecast_col": forecast_col,
            "order": str(pv_order),
            "seasonal_order": str(pv_seasonal_order),
            **fit_info,
            **sim_info,
            **quick_series_stats(e_scen.reshape(-1), f"pv_residual_scenario_bus_{bus}"),
        })

    fit_summary_df = pd.DataFrame(fit_rows)
    fit_summary_csv = OUT_DIR / "sarima_all_bus_fit_summary_69bus.csv"
    fit_summary_df.to_csv(fit_summary_csv, index=False)

    timestamp_str = dec_timestamps.astype(str).to_numpy()
    bus_numbers = np.arange(1, NUM_BUSES + 1, dtype=np.int32)

    np.savez_compressed(
        OUT_LOAD_SCENARIOS_NPZ,
        timestamps=timestamp_str,
        bus_numbers=bus_numbers,
        scenarios_kw=load_scenarios,
        residual_scenarios_kw=load_residual_scenarios,
        shape_note="scenarios_kw shape = [T_DEC, 69 buses, N_SCEN]",
    )

    np.savez_compressed(
        OUT_PV_SCENARIOS_NPZ,
        timestamps=timestamp_str,
        bus_numbers=bus_numbers,
        scenarios_kw=pv_scenarios,
        residual_scenarios_kw=pv_residual_scenarios,
        shape_note="scenarios_kw shape = [T_DEC, 69 buses, N_SCEN]",
    )

    fallback_summary = (
        fit_summary_df
        .groupby(["kind", "simulation_method"], dropna=False)
        .size()
        .reset_index(name="bus_count")
        .to_dict(orient="records")
    )

    manifest = {
        "input_wide_forecast_csv": str(WIDE_FORECAST_CSV),
        "input_residual_archive_csv": str(RESID_ARCHIVE_CSV),
        "output_dir": str(OUT_DIR),

        "num_buses": NUM_BUSES,
        "n_scen": N_SCEN,
        "seasonal_period": SEASONAL_PERIOD,
        "random_seed": RANDOM_SEED,

        "n_workers": N_WORKERS,
        "sarima_maxiter": SARIMA_MAXITER,
        "skip_grid_if_order_exists": SKIP_GRID_IF_ORDER_EXISTS,
        "force_grid_search": FORCE_GRID_SEARCH,
        "load_grid_was_run": load_grid_was_run,
        "pv_grid_was_run": pv_grid_was_run,

        "enforce_stationarity": ENFORCE_STATIONARITY,
        "enforce_invertibility": ENFORCE_INVERTIBILITY,
        "allow_nonconverged_final_fit": ALLOW_NONCONVERGED_FINAL_FIT,
        "bootstrap_block_size": BOOTSTRAP_BLOCK_SIZE,

        "representative_load_bus": rep_load_bus,
        "representative_load_col": rep_load_col,
        "representative_pv_bus": rep_pv_bus,
        "representative_pv_col": rep_pv_col,

        "load_order": list(load_order),
        "load_seasonal_order": list(load_seasonal_order),
        "pv_order": list(pv_order),
        "pv_seasonal_order": list(pv_seasonal_order),

        "load_scenarios_npz": str(OUT_LOAD_SCENARIOS_NPZ),
        "pv_scenarios_npz": str(OUT_PV_SCENARIOS_NPZ),
        "fit_summary_csv": str(fit_summary_csv),
        "fallback_summary": fallback_summary,

        "array_shape": {
            "load_scenarios": list(load_scenarios.shape),
            "pv_scenarios": list(pv_scenarios.shape),
            "meaning": "[T_DEC, 69 buses, N_SCEN]",
        },

        "long_csv_written": False,
    }

    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved load scenarios: {OUT_LOAD_SCENARIOS_NPZ}")
    print(f"Saved PV scenarios  : {OUT_PV_SCENARIOS_NPZ}")
    print(f"Fit summary         : {fit_summary_csv}")
    print(f"Manifest            : {OUT_SUMMARY_JSON}")
    print(f"Load shape          : {load_scenarios.shape}")
    print(f"PV shape            : {pv_scenarios.shape}")

    print("\nFallback / simulation summary:")
    print(pd.DataFrame(fallback_summary))


if __name__ == "__main__":
    main()
