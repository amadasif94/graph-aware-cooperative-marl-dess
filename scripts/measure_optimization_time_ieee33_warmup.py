#!/usr/bin/env python3
"""
Measure IEEE 33-bus deterministic MPC and stochastic MPC online
controller decision times using a fair steady-state timing protocol.

Timing protocol
---------------
For each controller:
    1. Load the IEEE33 forecast/scenario data.
    2. Run one untimed optimization warm-up solve.
    3. Reset the environment.
    4. Time all 96 rolling-horizon decisions in the selected episode.

Included in controller_decision_time_ms:
    - forecast/scenario-window extraction performed inside solve_*_action
    - price-window extraction
    - optimization-model construction
    - Gurobi optimization
    - solution extraction
    - first-action network-feasibility checking/correction

Excluded:
    - DESSEnv.step(...)
    - KPI calculation after the action is applied
    - CSV writing

The reported mean, median, and 95th percentile are therefore directly
comparable with RL actor inference times measured over the same 96-step
IEEE33 episode.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from experiments.mpc_smpc.optimization import baseline_config as cfg
from experiments.mpc_smpc.optimization.deterministic_mpc import (
    PointForecastData,
    load_point_forecast_data,
)
from experiments.mpc_smpc.optimization.scenario_data import (
    ScenarioData,
    load_scenario_data,
)
from experiments.mpc_smpc.optimization.rolling_horizon_eval import (
    build_env,
    build_step_row,
    build_timestamp_index_map,
    choose_available_horizon,
    normalize_topology_case,
    reset_at_episode_start,
    resolve_time_index,
    solve_mpc_action,
    solve_smpc_action,
)


def normalize_episodes(value: str) -> str | int:
    if str(value).lower() == "all":
        return "all"

    episodes = int(value)
    if episodes <= 0:
        raise ValueError("--episodes must be a positive integer or 'all'.")
    return episodes


def finite_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def percentile_95(series: pd.Series) -> float:
    values = finite_series(series)
    return float(values.quantile(0.95)) if not values.empty else float("nan")


def safe_sample_std(series: pd.Series) -> float:
    values = finite_series(series)
    return float(values.std(ddof=1)) if len(values) >= 2 else float("nan")


def verify_ieee33_imports_and_files() -> None:
    """Print and validate the exact Python modules and IEEE33 data files used."""
    import experiments.mpc_smpc.optimization.baseline_config as cfg_module
    import experiments.mpc_smpc.optimization.deterministic_mpc as mpc_module
    import experiments.mpc_smpc.optimization.scenario_data as scenario_module
    import experiments.mpc_smpc.optimization.rolling_horizon_eval as eval_module
    import experiments.mpc_smpc.optimization.stochastic_mpc as smpc_module

    module_paths = {
        "baseline_config": Path(cfg_module.__file__).resolve(),
        "deterministic_mpc": Path(mpc_module.__file__).resolve(),
        "scenario_data": Path(scenario_module.__file__).resolve(),
        "rolling_horizon_eval": Path(eval_module.__file__).resolve(),
        "stochastic_mpc": Path(smpc_module.__file__).resolve(),
    }

    print("=" * 80)
    print("IEEE33 IMPORT AND INPUT VERIFICATION")
    print("=" * 80)
    for name, path in module_paths.items():
        print(f"{name:24s}: {path}")

    print("-" * 80)
    print(f"NUM_BUSES               : {cfg.NUM_BUSES}")
    print(f"DESS_BUSES (zero-based) : {cfg.DESS_BUSES}")
    print(f"FORECAST_WIDE_FILE      : {Path(cfg.FORECAST_WIDE_FILE).resolve()}")
    print(f"LOAD_SCENARIO_FILE      : {Path(cfg.LOAD_SCENARIO_FILE).resolve()}")
    print(f"PV_SCENARIO_FILE        : {Path(cfg.PV_SCENARIO_FILE).resolve()}")
    print(f"Configured Gurobi threads: {getattr(cfg, 'THREADS', 'unknown')}")
    print("=" * 80)

    if int(cfg.NUM_BUSES) != 33:
        raise RuntimeError(
            f"Expected IEEE33 configuration, but NUM_BUSES={cfg.NUM_BUSES}."
        )

    expected_dess = [11, 15, 24, 29, 32]
    if list(cfg.DESS_BUSES) != expected_dess:
        raise RuntimeError(
            "Unexpected IEEE33 DESS placement. "
            f"Expected {expected_dess}, found {list(cfg.DESS_BUSES)}."
        )

    wrong_modules = [
        str(path) for path in module_paths.values()
        if path.name.endswith("_ieee69.py")
    ]
    if wrong_modules:
        raise RuntimeError(
            "IEEE69 modules were imported unexpectedly:\n" + "\n".join(wrong_modules)
        )

    required_files = [
        Path(cfg.FORECAST_WIDE_FILE),
        Path(cfg.LOAD_SCENARIO_FILE),
        Path(cfg.PV_SCENARIO_FILE),
    ]
    missing = [str(path.resolve()) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required IEEE33 input files were not found:\n" + "\n".join(missing)
        )


def load_controller_data(
    controller: str,
    n_scenarios: int,
) -> tuple[Optional[ScenarioData], Optional[PointForecastData]]:
    if controller == "smpc":
        scenario_data = load_scenario_data(
            max_steps=cfg.MAX_STEPS,
            max_scenarios=max(int(n_scenarios), 1),
        )
        return scenario_data, None

    if controller == "mpc":
        point_data = load_point_forecast_data(max_steps=cfg.MAX_STEPS)
        return None, point_data

    raise ValueError(f"Unknown controller: {controller}")


def build_data_time_maps(
    scenario_data: Optional[ScenarioData],
    point_data: Optional[PointForecastData],
):
    scenario_time_map = (
        build_timestamp_index_map(scenario_data.timestamps)
        if scenario_data is not None
        else None
    )
    point_time_map = (
        build_timestamp_index_map(point_data.timestamps)
        if point_data is not None
        else None
    )
    return scenario_time_map, point_time_map


def solve_one_action(
    controller: str,
    env,
    current_date_time,
    scenario_data: Optional[ScenarioData],
    point_data: Optional[PointForecastData],
    scenario_time_map,
    point_time_map,
    horizon_t: int,
    n_scenarios: int,
    topology_case: str,
) -> tuple[Dict[str, Any], int, int]:
    """Solve one MPC/SMPC action and return result, data index, and horizon."""
    if controller == "smpc":
        if scenario_data is None or scenario_time_map is None:
            raise ValueError("Scenario data is required for SMPC.")

        data_time_index = resolve_time_index(current_date_time, scenario_time_map)
        available_horizon = choose_available_horizon(
            requested_horizon=horizon_t,
            data_index=data_time_index,
            data_length=len(scenario_data.timestamps),
            env_current_index=env.current_index,
            env_total_length=len(env.time_series["df"]),
        )

        solve_result = solve_smpc_action(
            scenario_data=scenario_data,
            env=env,
            scenario_time_index=data_time_index,
            horizon_t=available_horizon,
            n_scenarios=n_scenarios,
            topology_case=topology_case,
        )
        return solve_result, int(data_time_index), int(available_horizon)

    if controller == "mpc":
        if point_data is None or point_time_map is None:
            raise ValueError("Point-forecast data is required for MPC.")

        data_time_index = resolve_time_index(current_date_time, point_time_map)
        available_horizon = choose_available_horizon(
            requested_horizon=horizon_t,
            data_index=data_time_index,
            data_length=len(point_data.timestamps),
            env_current_index=env.current_index,
            env_total_length=len(env.time_series["df"]),
        )

        solve_result = solve_mpc_action(
            point_data=point_data,
            env=env,
            forecast_time_index=data_time_index,
            horizon_t=available_horizon,
            topology_case=topology_case,
        )
        return solve_result, int(data_time_index), int(available_horizon)

    raise ValueError("controller must be either 'mpc' or 'smpc'.")


def run_untimed_warmup(
    controller: str,
    env,
    scenario_data: Optional[ScenarioData],
    point_data: Optional[PointForecastData],
    start_index: int,
    seed: int,
    horizon_t: int,
    n_scenarios: int,
    topology_case: str,
) -> None:
    """Run one untimed solve to initialize Gurobi and controller internals."""
    scenario_time_map, point_time_map = build_data_time_maps(
        scenario_data=scenario_data,
        point_data=point_data,
    )

    _, initial_info = reset_at_episode_start(
        env=env,
        start_index=int(start_index),
        seed=seed,
    )

    print("-" * 80)
    print(
        f"Untimed {controller.upper()} warm-up solve | "
        f"start_index={start_index} | date={initial_info['date_time']}"
    )

    warmup_start = time.perf_counter()
    solve_result, _, _ = solve_one_action(
        controller=controller,
        env=env,
        current_date_time=initial_info["date_time"],
        scenario_data=scenario_data,
        point_data=point_data,
        scenario_time_map=scenario_time_map,
        point_time_map=point_time_map,
        horizon_t=horizon_t,
        n_scenarios=n_scenarios,
        topology_case=topology_case,
    )
    warmup_elapsed = time.perf_counter() - warmup_start

    status = str(solve_result.get("status", "unknown"))
    if "action" not in solve_result:
        raise RuntimeError("Warm-up solve did not return an action.")

    print(
        f"Warm-up complete | status={status} | "
        f"elapsed={warmup_elapsed:.3f} s | excluded from all summaries"
    )
    print("-" * 80)


def evaluate_timed_episode(
    controller: str,
    env,
    scenario_data: Optional[ScenarioData],
    point_data: Optional[PointForecastData],
    episode_id: int,
    start_index: int,
    seed: int,
    horizon_t: int,
    n_scenarios: int,
    topology_case: str,
    control_interval_s: float,
    print_every: int,
) -> list[Dict[str, Any]]:
    controller = str(controller).lower()
    topology_case = normalize_topology_case(topology_case)

    _, initial_info = reset_at_episode_start(
        env=env,
        start_index=start_index,
        seed=seed,
    )

    scenario_time_map, point_time_map = build_data_time_maps(
        scenario_data=scenario_data,
        point_data=point_data,
    )

    rows: list[Dict[str, Any]] = []
    done = False
    step = 0
    current_date_time = initial_info["date_time"]

    while not done:
        decision_start_ns = time.perf_counter_ns()

        solve_result, data_time_index, available_horizon = solve_one_action(
            controller=controller,
            env=env,
            current_date_time=current_date_time,
            scenario_data=scenario_data,
            point_data=point_data,
            scenario_time_map=scenario_time_map,
            point_time_map=point_time_map,
            horizon_t=horizon_t,
            n_scenarios=n_scenarios,
            topology_case=topology_case,
        )

        decision_end_ns = time.perf_counter_ns()
        controller_decision_time_s = (
            decision_end_ns - decision_start_ns
        ) / 1_000_000_000.0

        requested_action = np.asarray(solve_result["action"], dtype=np.float32)

        # Environment transition is intentionally outside the timing boundary.
        _, reward, terminated, truncated, info = env.step(requested_action)
        done = bool(terminated or truncated)

        row = build_step_row(
            controller=controller,
            topology_case=topology_case,
            episode_id=episode_id,
            step=step,
            start_index=start_index,
            scenario_time_index=data_time_index,
            requested_action=requested_action,
            solve_result=solve_result,
            reward=reward,
            info=info,
            env=env,
        )

        solver_time_s = float(solve_result.get("solve_time_sec", np.nan))
        row.update(
            {
                "warmup_excluded": True,
                "requested_horizon": int(horizon_t),
                "available_horizon": int(available_horizon),
                "n_scenarios": int(n_scenarios if controller == "smpc" else 1),
                "controller_decision_time_s": float(controller_decision_time_s),
                "controller_decision_time_ms": float(
                    controller_decision_time_s * 1000.0
                ),
                "solver_time_s": solver_time_s,
                "solver_time_ms": (
                    float(solver_time_s * 1000.0)
                    if np.isfinite(solver_time_s)
                    else np.nan
                ),
                "non_solver_overhead_time_ms": (
                    float((controller_decision_time_s - solver_time_s) * 1000.0)
                    if np.isfinite(solver_time_s)
                    else np.nan
                ),
                "control_interval_s": float(control_interval_s),
                "time_fraction_of_control_interval": float(
                    controller_decision_time_s / control_interval_s
                ),
            }
        )
        rows.append(row)

        should_print = (
            step == 0
            or ((step + 1) % int(print_every) == 0)
            or done
        )
        if should_print:
            solver_ms_text = (
                f"{row['solver_time_ms']:.3f}"
                if np.isfinite(row["solver_time_ms"])
                else "nan"
            )
            print(
                f"topology={topology_case} "
                f"controller={controller} "
                f"episode={episode_id} "
                f"step={step + 1} "
                f"date={info['date_time']} "
                f"decision_ms={row['controller_decision_time_ms']:.3f} "
                f"solver_ms={solver_ms_text} "
                f"status={row['solver_status']} "
                f"feasible={row['feasible']}"
            )

        current_date_time = env.last_date_time
        step += 1

    return rows


def build_episode_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    if step_df.empty:
        return pd.DataFrame()

    rows = []
    group_columns = ["topology_case", "controller", "episode_id", "start_index"]

    for keys, group in step_df.groupby(group_columns, sort=False):
        topology_case, controller, episode_id, start_index = keys
        decision_ms = finite_series(group["controller_decision_time_ms"])
        solver_ms = finite_series(group["solver_time_ms"])
        overhead_ms = finite_series(group["non_solver_overhead_time_ms"])

        rows.append(
            {
                "topology_case": topology_case,
                "controller": controller,
                "episode_id": int(episode_id),
                "start_index": int(start_index),
                "steps": int(len(group)),
                "requested_horizon": int(group["requested_horizon"].iloc[0]),
                "n_scenarios": int(group["n_scenarios"].iloc[0]),
                "mean_decision_time_ms": float(decision_ms.mean()),
                "std_decision_time_ms": safe_sample_std(decision_ms),
                "median_decision_time_ms": float(decision_ms.median()),
                "p95_decision_time_ms": percentile_95(decision_ms),
                "min_decision_time_ms": float(decision_ms.min()),
                "max_decision_time_ms": float(decision_ms.max()),
                "total_episode_decision_time_ms": float(decision_ms.sum()),
                "mean_solver_time_ms": (
                    float(solver_ms.mean()) if not solver_ms.empty else np.nan
                ),
                "std_solver_time_ms": safe_sample_std(solver_ms),
                "median_solver_time_ms": (
                    float(solver_ms.median()) if not solver_ms.empty else np.nan
                ),
                "p95_solver_time_ms": percentile_95(solver_ms),
                "min_solver_time_ms": (
                    float(solver_ms.min()) if not solver_ms.empty else np.nan
                ),
                "max_solver_time_ms": (
                    float(solver_ms.max()) if not solver_ms.empty else np.nan
                ),
                "mean_non_solver_overhead_time_ms": (
                    float(overhead_ms.mean()) if not overhead_ms.empty else np.nan
                ),
                "control_interval_s": float(group["control_interval_s"].iloc[0]),
                "mean_time_fraction_of_control_interval": float(
                    group["time_fraction_of_control_interval"].mean()
                ),
                "feasible_rate": float(group["feasible"].mean()),
                "converged_rate": float(group["converged"].mean()),
                "solver_success_rate": float(
                    group["solver_status"]
                    .astype(str)
                    .str.lower()
                    .isin(["optimal", "suboptimal", "time_limit"])
                    .mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_aggregate_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in episode_df.groupby(
        ["topology_case", "controller"], sort=False
    ):
        topology_case, controller = keys
        rows.append(
            {
                "topology_case": topology_case,
                "controller": controller,
                "episodes": int(len(group)),
                "steps": int(group["steps"].sum()),
                "requested_horizon": int(group["requested_horizon"].iloc[0]),
                "n_scenarios": int(group["n_scenarios"].iloc[0]),
                "mean_decision_time_ms": float(
                    group["mean_decision_time_ms"].mean()
                ),
                "std_across_episode_means_ms": safe_sample_std(
                    group["mean_decision_time_ms"]
                ),
                "mean_median_decision_time_ms": float(
                    group["median_decision_time_ms"].mean()
                ),
                "mean_p95_decision_time_ms": float(
                    group["p95_decision_time_ms"].mean()
                ),
                "worst_decision_time_ms": float(
                    group["max_decision_time_ms"].max()
                ),
                "mean_solver_time_ms": float(
                    group["mean_solver_time_ms"].mean()
                ),
                "mean_p95_solver_time_ms": float(
                    group["p95_solver_time_ms"].mean()
                ),
                "worst_solver_time_ms": float(
                    group["max_solver_time_ms"].max()
                ),
                "mean_non_solver_overhead_time_ms": float(
                    group["mean_non_solver_overhead_time_ms"].mean()
                ),
                "control_interval_s": float(group["control_interval_s"].iloc[0]),
                "mean_time_fraction_of_control_interval": float(
                    group["mean_time_fraction_of_control_interval"].mean()
                ),
                "mean_feasible_rate": float(group["feasible_rate"].mean()),
                "mean_converged_rate": float(group["converged_rate"].mean()),
                "mean_solver_success_rate": float(
                    group["solver_success_rate"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_timing_evaluation(
    controller: str,
    episodes: str | int,
    horizon_t: int,
    n_scenarios: int,
    seed: int,
    mode: str,
    topology_case: str,
    control_interval_s: float,
    print_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    controller = str(controller).lower()
    topology_case = normalize_topology_case(topology_case)

    env = build_env(mode=mode, seed=seed, topology_case=topology_case)
    scenario_data, point_data = load_controller_data(
        controller=controller,
        n_scenarios=n_scenarios,
    )

    if episodes == "all":
        start_indices = list(env.episode_start_indices)
    else:
        start_indices = list(env.episode_start_indices[: int(episodes)])

    if not start_indices:
        raise ValueError("No episode start indices are available.")

    print("=" * 80)
    print("IEEE33 OPTIMIZATION CONTROLLER TIMING")
    print("=" * 80)
    print(f"Controller          : {controller}")
    print(f"Topology            : {topology_case}")
    print(f"Mode                : {mode}")
    print(f"Episodes            : {len(start_indices)}")
    print(f"Horizon             : {horizon_t}")
    print(f"Scenarios           : {n_scenarios if controller == 'smpc' else 1}")
    print(f"Control interval s  : {control_interval_s}")
    print(f"Configured threads  : {getattr(cfg, 'THREADS', 'unknown')}")
    print("=" * 80)

    # Exactly one untimed warm-up solve per controller.
    run_untimed_warmup(
        controller=controller,
        env=env,
        scenario_data=scenario_data,
        point_data=point_data,
        start_index=int(start_indices[0]),
        seed=seed,
        horizon_t=horizon_t,
        n_scenarios=n_scenarios,
        topology_case=topology_case,
    )

    all_rows: list[Dict[str, Any]] = []
    for episode_id, start_index in enumerate(start_indices):
        print()
        print("-" * 80)
        print(
            f"Controller {controller.upper()} | "
            f"Episode {episode_id + 1}/{len(start_indices)} | "
            f"start_index={start_index}"
        )
        print("-" * 80)

        episode_rows = evaluate_timed_episode(
            controller=controller,
            env=env,
            scenario_data=scenario_data,
            point_data=point_data,
            episode_id=episode_id,
            start_index=int(start_index),
            seed=seed + episode_id,
            horizon_t=horizon_t,
            n_scenarios=n_scenarios,
            topology_case=topology_case,
            control_interval_s=control_interval_s,
            print_every=print_every,
        )
        all_rows.extend(episode_rows)

    step_df = pd.DataFrame(all_rows)
    episode_df = build_episode_summary(step_df)
    aggregate_df = build_aggregate_summary(episode_df)
    return step_df, episode_df, aggregate_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure IEEE33 MPC/SMPC steady-state online decision latency."
    )
    parser.add_argument(
        "--controller",
        choices=["mpc", "smpc", "both"],
        default="both",
    )
    parser.add_argument("--topology", type=str, default="TP1")
    parser.add_argument(
        "--episodes",
        type=str,
        default="1",
        help="Positive integer or 'all'.",
    )
    parser.add_argument("--horizon", type=int, default=cfg.HORIZON_T)
    parser.add_argument(
        "--scenarios",
        type=int,
        default=10,
        help="Scenario count for SMPC; ignored by MPC.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument(
        "--control_interval_s",
        type=float,
        default=900.0,
    )
    parser.add_argument("--print_every", type=int, default=24)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/timing/IEEE33_optimization_warmup",
    )

    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive.")
    if args.scenarios <= 0:
        parser.error("--scenarios must be positive.")
    if args.control_interval_s <= 0.0:
        parser.error("--control_interval_s must be positive.")
    if args.print_every <= 0:
        parser.error("--print_every must be positive.")
    return args


def main() -> None:
    args = parse_args()
    episodes = normalize_episodes(args.episodes)
    topology_case = normalize_topology_case(args.topology)

    verify_ieee33_imports_and_files()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    controllers = (
        ["mpc", "smpc"] if args.controller == "both" else [args.controller]
    )

    print("=" * 80)
    print("IEEE33 MPC/SMPC COMPUTATIONAL-EFFICIENCY EVALUATION")
    print("=" * 80)
    print(f"Controllers         : {controllers}")
    print(f"Topology            : {topology_case}")
    print(f"Episodes            : {episodes}")
    print(f"Horizon             : {args.horizon}")
    print(f"SMPC scenarios      : {args.scenarios}")
    print(f"Mode                : {args.mode}")
    print(f"Output directory    : {out_base}")
    print(f"Python              : {platform.python_version()}")
    print(f"Platform            : {platform.platform()}")
    print(f"Configured threads  : {getattr(cfg, 'THREADS', 'unknown')}")
    print("Warm-up policy      : one untimed solve per controller")
    print("=" * 80)

    combined_step_frames = []
    combined_episode_frames = []
    combined_aggregate_frames = []

    for controller in controllers:
        controller_dir = out_base / controller
        controller_dir.mkdir(parents=True, exist_ok=True)

        step_df, episode_df, aggregate_df = run_timing_evaluation(
            controller=controller,
            episodes=episodes,
            horizon_t=args.horizon,
            n_scenarios=args.scenarios,
            seed=args.seed,
            mode=args.mode,
            topology_case=topology_case,
            control_interval_s=args.control_interval_s,
            print_every=args.print_every,
        )

        step_path = controller_dir / "timing_per_step.csv"
        episode_path = controller_dir / "timing_episode_summary.csv"
        aggregate_path = controller_dir / "timing_aggregate_summary.csv"

        step_df.to_csv(step_path, index=False)
        episode_df.to_csv(episode_path, index=False)
        aggregate_df.to_csv(aggregate_path, index=False)

        combined_step_frames.append(step_df)
        combined_episode_frames.append(episode_df)
        combined_aggregate_frames.append(aggregate_df)

        print()
        print("=" * 80)
        print(f"{controller.upper()} TIMING SUMMARY")
        print("=" * 80)
        print(aggregate_df.to_string(index=False))
        print(f"Saved per-step timing      : {step_path}")
        print(f"Saved episode summary      : {episode_path}")
        print(f"Saved aggregate summary    : {aggregate_path}")

    combined_step_df = pd.concat(combined_step_frames, ignore_index=True)
    combined_episode_df = pd.concat(combined_episode_frames, ignore_index=True)
    combined_aggregate_df = pd.concat(combined_aggregate_frames, ignore_index=True)

    combined_step_path = out_base / "combined_optimization_timing_per_step.csv"
    combined_episode_path = (
        out_base / "combined_optimization_timing_episode_summary.csv"
    )
    combined_aggregate_path = (
        out_base / "combined_optimization_timing_summary.csv"
    )

    combined_step_df.to_csv(combined_step_path, index=False)
    combined_episode_df.to_csv(combined_episode_path, index=False)
    combined_aggregate_df.to_csv(combined_aggregate_path, index=False)

    print()
    print("=" * 80)
    print("COMBINED MPC/SMPC TIMING SUMMARY")
    print("=" * 80)
    print(combined_aggregate_df.to_string(index=False))
    print(f"Saved combined per-step data : {combined_step_path}")
    print(f"Saved combined episodes      : {combined_episode_path}")
    print(f"Saved combined summary       : {combined_aggregate_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
