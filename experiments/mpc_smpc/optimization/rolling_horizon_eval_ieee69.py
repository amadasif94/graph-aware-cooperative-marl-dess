# ============================================================
# rolling_horizon_eval_ieee69.py
# ============================================================

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv
from experiments.topology_generalization.topology_utils_ieee69 import (
    build_topology_config,
)

from experiments.mpc_smpc.optimization import baseline_config_ieee69 as cfg

from experiments.mpc_smpc.optimization.scenario_data_ieee69 import (
    ScenarioData,
    load_scenario_data,
    get_nodal_smpc_window,
)

from experiments.mpc_smpc.optimization.deterministic_mpc_ieee69 import (
    PointForecastData,
    load_point_forecast_data,
    get_point_forecast_window,
    solve_deterministic_mpc,
)

from experiments.mpc_smpc.optimization.stochastic_mpc_ieee69 import (
    build_stochastic_mpc_problem,
    solve_stochastic_mpc,
)


# ============================================================
# Topology / Environment helpers
# ============================================================

def normalize_topology_case(topology_case: str = "TP1") -> str:
    if hasattr(cfg, "normalize_topology_case"):
        return cfg.normalize_topology_case(topology_case)
    return str(topology_case or "TP1").upper()


def build_topology_env_config(topology_case: str = "TP1") -> dict:
    topology_case = normalize_topology_case(topology_case)

    if topology_case == "TP1":
        return copy.deepcopy(IEEE69_CONFIG)

    return build_topology_config(
        case_name=topology_case,
        base_config=copy.deepcopy(IEEE69_CONFIG),
    )


def build_env(
    mode: str = "test",
    seed: int = 42,
    topology_case: str = "TP1",
) -> DESSEnv:
    config = build_topology_env_config(topology_case)
    return DESSEnv(config=config, mode=mode, seed=seed)


def reset_at_episode_start(env: DESSEnv, start_index: int, seed: int):
    old_starts = env.episode_start_indices
    env.episode_start_indices = [int(start_index)]

    obs, info = env.reset(seed=seed)

    env.episode_start_indices = old_starts
    return obs, info


def get_soc_vector(env: DESSEnv) -> np.ndarray:
    return np.asarray([b.get_soc() for b in env.batteries], dtype=np.float64)


def build_timestamp_index_map(timestamps: pd.DatetimeIndex) -> Dict[Any, int]:
    return {
        pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None): i
        for i, ts in enumerate(pd.DatetimeIndex(timestamps))
    }


def resolve_time_index(date_time, timestamp_map: Dict[Any, int]) -> int:
    key = pd.Timestamp(date_time).to_pydatetime().replace(tzinfo=None)

    if key not in timestamp_map:
        raise KeyError(f"Timestamp {key} not found in forecast/scenario data.")

    return int(timestamp_map[key])


def get_env_price_window(
    env: DESSEnv,
    current_index: int,
    horizon_t: int,
) -> np.ndarray:
    prices = []

    for k in range(int(horizon_t)):
        idx = int(current_index) + k
        _, _, _, price, _ = env._get_profiles(idx)
        prices.append(float(price))

    return np.asarray(prices, dtype=np.float64)


def choose_available_horizon(
    requested_horizon: int,
    data_index: int,
    data_length: int,
    env_current_index: int,
    env_total_length: int,
) -> int:
    h = min(
        int(requested_horizon),
        int(data_length) - int(data_index),
        int(env_total_length) - int(env_current_index),
    )

    if h <= 0:
        raise ValueError("No valid horizon available.")

    return int(h)


# ============================================================
# Metrics
# ============================================================

def build_step_row(
    controller: str,
    topology_case: str,
    episode_id: int,
    step: int,
    start_index: int,
    scenario_time_index: int,
    requested_action: np.ndarray,
    solve_result: Dict[str, Any],
    reward: np.ndarray,
    info: Dict[str, Any],
    env: DESSEnv,
) -> Dict[str, Any]:

    topology_case = normalize_topology_case(topology_case)

    reward = np.asarray(reward, dtype=np.float64).reshape(-1)
    requested_action = np.asarray(requested_action, dtype=np.float64).reshape(-1)
    accepted_action = np.asarray(info["accepted_action"], dtype=np.float64).reshape(-1)

    kpis = info["kpis"]

    dess_power_kw_full = np.asarray(info["dess_power_kw"], dtype=np.float64)
    dess_power_agents = dess_power_kw_full[env.dess_buses]

    grid_import_kw = float(kpis["grid_import_kw"])
    curtailment_kw = float(kpis["curtailment_kw"])
    price = float(kpis["price"])

    grid_import_mwh = max(0.0, grid_import_kw) * env.delta_t_hours / 1000.0
    curtailment_mwh = curtailment_kw * env.delta_t_hours / 1000.0
    throughput_mwh = (
        float(np.sum(np.abs(dess_power_agents)))
        * env.delta_t_hours
        / 1000.0
    )
    energy_cost = price * grid_import_mwh

    soc_values = [float(b.get_soc()) for b in env.batteries]

    row = {
        "system": "ieee69",
        "topology_case": topology_case,
        "controller": str(controller),
        "episode_id": int(episode_id),
        "step": int(step),
        "date_time": info["date_time"],
        "start_index": int(start_index),
        "scenario_time_index": int(scenario_time_index),

        "reward_mean": float(np.mean(reward)),
        "reward_sum": float(np.sum(reward)),

        "grid_import_kw": grid_import_kw,
        "grid_import_mwh": grid_import_mwh,
        "energy_cost": energy_cost,

        "curtailment_kw": curtailment_kw,
        "curtailment_mwh": curtailment_mwh,

        "voltage_deviation": float(kpis["voltage_deviation"]),
        "grid_stress": float(kpis["grid_stress"]),

        "min_voltage_pu": float(info["min_voltage_pu"]),
        "max_voltage_pu": float(info["max_voltage_pu"]),
        "max_line_current_pu": float(info["max_line_current_pu"]),

        "max_voltage_violation": float(info["max_voltage_violation"]),
        "max_line_current_violation": float(info["max_line_current_violation"]),

        "feasible": bool(info["feasible"]),
        "converged": bool(info["converged"]),
        "infeasible_action": bool(info["infeasible_action"]),

        "throughput_mwh": throughput_mwh,

        "solver_status": str(solve_result.get("status", "none")),
        "objective_value": float(solve_result.get("objective_value", np.nan)),
        "solve_time_sec": float(solve_result.get("solve_time_sec", np.nan)),
    }

    network_check = solve_result.get("network_check", {})

    if isinstance(network_check, dict):
        row["smpc_network_feasible"] = bool(network_check.get("network_feasible", True))
        row["smpc_network_corrected"] = bool(network_check.get("corrected", False))
        row["smpc_correction_attempts"] = int(network_check.get("correction_attempts", 0))
        row["smpc_check_topology_case"] = str(network_check.get("topology_case", topology_case))
        row["smpc_check_worst_min_voltage_pu"] = float(network_check.get("worst_min_voltage_pu", np.nan))
        row["smpc_check_worst_max_voltage_pu"] = float(network_check.get("worst_max_voltage_pu", np.nan))
        row["smpc_check_worst_line_current_pu"] = float(network_check.get("worst_line_current_pu", np.nan))
        row["smpc_check_max_voltage_violation"] = float(network_check.get("max_voltage_violation", np.nan))
        row["smpc_check_max_line_current_violation"] = float(network_check.get("max_line_current_violation", np.nan))

    beta_ch = solve_result.get("beta_ch_kw", None)
    beta_dis = solve_result.get("beta_dis_kw", None)

    if beta_ch is not None:
        beta_ch = np.asarray(beta_ch, dtype=np.float64).reshape(-1)

    if beta_dis is not None:
        beta_dis = np.asarray(beta_dis, dtype=np.float64).reshape(-1)

    for i in range(env.num_agents):
        row[f"requested_action_agent_{i}"] = float(requested_action[i])
        row[f"accepted_action_agent_{i}"] = float(accepted_action[i])
        row[f"dess_power_kw_agent_{i}"] = float(dess_power_agents[i])
        row[f"soc_agent_{i}"] = float(soc_values[i])

        if beta_ch is not None and i < len(beta_ch):
            row[f"beta_ch_kw_agent_{i}"] = float(beta_ch[i])

        if beta_dis is not None and i < len(beta_dis):
            row[f"beta_dis_kw_agent_{i}"] = float(beta_dis[i])

    return row


def summarize_results(step_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if step_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    episode_summary = (
        step_df
        .groupby(["system", "topology_case", "controller", "episode_id", "start_index"], as_index=False)
        .agg(
            steps=("step", "count"),
            total_reward_mean=("reward_mean", "sum"),
            total_reward_team=("reward_sum", "sum"),
            energy_cost=("energy_cost", "sum"),
            grid_import_mwh=("grid_import_mwh", "sum"),
            curtailment_mwh=("curtailment_mwh", "sum"),
            throughput_mwh=("throughput_mwh", "sum"),
            mean_voltage_deviation=("voltage_deviation", "mean"),
            worst_min_voltage_pu=("min_voltage_pu", "min"),
            worst_max_voltage_pu=("max_voltage_pu", "max"),
            worst_line_current_pu=("max_line_current_pu", "max"),
            worst_voltage_violation=("max_voltage_violation", "max"),
            worst_line_current_violation=("max_line_current_violation", "max"),
            mean_feasible_rate=("feasible", "mean"),
            mean_converged_rate=("converged", "mean"),
            mean_infeasible_requested_count=("infeasible_action", "sum"),
            avg_solve_time_sec=("solve_time_sec", "mean"),
            max_solve_time_sec=("solve_time_sec", "max"),
        )
    )

    aggregate_summary = (
        episode_summary
        .groupby(["system", "topology_case", "controller"], as_index=False)
        .agg(
            episodes=("episode_id", "count"),
            mean_total_reward_mean=("total_reward_mean", "mean"),
            std_total_reward_mean=("total_reward_mean", "std"),
            mean_total_reward_team=("total_reward_team", "mean"),
            std_total_reward_team=("total_reward_team", "std"),
            mean_energy_cost=("energy_cost", "mean"),
            mean_grid_import_mwh=("grid_import_mwh", "mean"),
            mean_curtailment_mwh=("curtailment_mwh", "mean"),
            mean_throughput_mwh=("throughput_mwh", "mean"),
            mean_voltage_deviation=("mean_voltage_deviation", "mean"),
            worst_min_voltage_pu=("worst_min_voltage_pu", "min"),
            worst_max_voltage_pu=("worst_max_voltage_pu", "max"),
            worst_line_current_pu=("worst_line_current_pu", "max"),
            worst_voltage_violation=("worst_voltage_violation", "max"),
            worst_line_current_violation=("worst_line_current_violation", "max"),
            mean_infeasible_requested_count=("mean_infeasible_requested_count", "mean"),
            mean_feasible_rate=("mean_feasible_rate", "mean"),
            mean_converged_rate=("mean_converged_rate", "mean"),
            avg_solve_time_sec=("avg_solve_time_sec", "mean"),
            max_solve_time_sec=("max_solve_time_sec", "max"),
        )
    )

    return episode_summary, aggregate_summary


# ============================================================
# Controller solve dispatch
# ============================================================

def solve_smpc_action(
    scenario_data: ScenarioData,
    env: DESSEnv,
    scenario_time_index: int,
    horizon_t: int,
    n_scenarios: int,
    topology_case: str,
) -> Dict[str, Any]:

    topology_case = normalize_topology_case(topology_case)

    nodal = get_nodal_smpc_window(
        data=scenario_data,
        t0=scenario_time_index,
        horizon_t=horizon_t,
        n_scenarios=n_scenarios,
    )

    price_T = get_env_price_window(
        env=env,
        current_index=env.current_index,
        horizon_t=horizon_t,
    )

    bundle = build_stochastic_mpc_problem(
        S=int(n_scenarios),
        T=int(horizon_t),
    )

    return solve_stochastic_mpc(
        bundle=bundle,
        aggregate_netload_ST_kw=None,
        load_ST33_kw=nodal["load_kw"],
        pv_ST33_kw=nodal["pv_kw"],
        price_T=price_T,
        soc0=get_soc_vector(env),
        enforce_network_first_action=True,
        topology_config=env.config,
        topology_case=topology_case,
    )


def solve_mpc_action(
    point_data: PointForecastData,
    env: DESSEnv,
    forecast_time_index: int,
    horizon_t: int,
    topology_case: str,
) -> Dict[str, Any]:

    topology_case = normalize_topology_case(topology_case)

    win = get_point_forecast_window(
        data=point_data,
        t0=forecast_time_index,
        horizon_t=horizon_t,
    )

    price_T = get_env_price_window(
        env=env,
        current_index=env.current_index,
        horizon_t=horizon_t,
    )

    return solve_deterministic_mpc(
        aggregate_netload_T_kw=win["aggregate_netload_kw"],
        price_T=price_T,
        soc0=get_soc_vector(env),
        load_T33_kw=win["load_kw"],
        pv_T33_kw=win["pv_kw"],
        topology_config=env.config,
        topology_case=topology_case,
        enforce_network_first_action=True,
    )


# ============================================================
# Episode evaluation
# ============================================================

def evaluate_one_episode(
    controller: str,
    env: DESSEnv,
    scenario_data: Optional[ScenarioData],
    point_data: Optional[PointForecastData],
    episode_id: int,
    start_index: int,
    seed: int,
    horizon_t: int,
    n_scenarios: int,
    topology_case: str = "TP1",
    print_every: int = 24,
) -> list[Dict[str, Any]]:

    controller = str(controller).lower()
    topology_case = normalize_topology_case(topology_case)

    _, info0 = reset_at_episode_start(
        env=env,
        start_index=start_index,
        seed=seed,
    )

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

    rows = []
    done = False
    step = 0
    current_date_time = info0["date_time"]

    while not done:

        if controller == "smpc":
            if scenario_data is None:
                raise ValueError("scenario_data is required for SMPC.")

            scenario_time_index = resolve_time_index(
                current_date_time,
                scenario_time_map,
            )

            h = choose_available_horizon(
                requested_horizon=horizon_t,
                data_index=scenario_time_index,
                data_length=len(scenario_data.timestamps),
                env_current_index=env.current_index,
                env_total_length=len(env.time_series["df"]),
            )

            solve_result = solve_smpc_action(
                scenario_data=scenario_data,
                env=env,
                scenario_time_index=scenario_time_index,
                horizon_t=h,
                n_scenarios=n_scenarios,
                topology_case=topology_case,
            )

        elif controller in ["mpc", "deterministic_mpc"]:
            if point_data is None:
                raise ValueError("point_data is required for deterministic MPC.")

            scenario_time_index = resolve_time_index(
                current_date_time,
                point_time_map,
            )

            h = choose_available_horizon(
                requested_horizon=horizon_t,
                data_index=scenario_time_index,
                data_length=len(point_data.timestamps),
                env_current_index=env.current_index,
                env_total_length=len(env.time_series["df"]),
            )

            solve_result = solve_mpc_action(
                point_data=point_data,
                env=env,
                forecast_time_index=scenario_time_index,
                horizon_t=h,
                topology_case=topology_case,
            )

            controller = "mpc"

        else:
            raise ValueError("controller must be 'smpc' or 'mpc'.")

        requested_action = np.asarray(solve_result["action"], dtype=np.float32)

        _, reward, terminated, truncated, info = env.step(requested_action)
        done = bool(terminated or truncated)

        row = build_step_row(
            controller=controller,
            topology_case=topology_case,
            episode_id=episode_id,
            step=step,
            start_index=start_index,
            scenario_time_index=scenario_time_index,
            requested_action=requested_action,
            solve_result=solve_result,
            reward=reward,
            info=info,
            env=env,
        )

        rows.append(row)

        if step == 0 or ((step + 1) % int(print_every) == 0) or done:
            print(
                f"system=ieee69 "
                f"topology={topology_case} "
                f"controller={controller} "
                f"episode={episode_id} "
                f"step={step + 1} "
                f"date={info['date_time']} "
                f"reward_sum={row['reward_sum']:.4f} "
                f"cost={row['energy_cost']:.4f} "
                f"feasible={row['feasible']} "
                f"status={row['solver_status']}"
            )

        current_date_time = env.last_date_time
        step += 1

    return rows


# ============================================================
# Evaluation runner
# ============================================================

def _output_paths(
    controller: str,
    output_dir: Path,
    horizon_t: int,
    n_scenarios: int,
    topology_case: str,
):
    topology_case = normalize_topology_case(topology_case)

    if controller == "mpc":
        return (
            output_dir / f"step_metrics_mpc_ieee69_{topology_case}_h{horizon_t}.csv",
            output_dir / f"episode_summary_mpc_ieee69_{topology_case}_h{horizon_t}.csv",
            output_dir / f"aggregate_summary_mpc_ieee69_{topology_case}_h{horizon_t}.csv",
        )

    return (
        output_dir / f"step_metrics_smpc_ieee69_{topology_case}_h{horizon_t}_S{n_scenarios}.csv",
        output_dir / f"episode_summary_smpc_ieee69_{topology_case}_h{horizon_t}_S{n_scenarios}.csv",
        output_dir / f"aggregate_summary_smpc_ieee69_{topology_case}_h{horizon_t}_S{n_scenarios}.csv",
    )


def run_controller_evaluation(
    controller: str,
    episodes: str | int = 1,
    horizon_t: int = cfg.HORIZON_T,
    n_scenarios: int = 10,
    seed: int = 42,
    mode: str = "test",
    topology_case: str = "TP1",
    output_dir: Optional[Path] = None,
    print_every: int = 24,
) -> Dict[str, pd.DataFrame]:

    controller = str(controller).lower()
    topology_case = normalize_topology_case(topology_case)

    env = build_env(
        mode=mode,
        seed=seed,
        topology_case=topology_case,
    )

    scenario_data = None
    point_data = None

    if controller == "smpc":
        scenario_data = load_scenario_data(
            max_steps=cfg.MAX_STEPS,
            max_scenarios=max(n_scenarios, 1),
        )

    elif controller in ["mpc", "deterministic_mpc"]:
        point_data = load_point_forecast_data(
            max_steps=cfg.MAX_STEPS,
        )
        controller = "mpc"

    else:
        raise ValueError("controller must be 'smpc' or 'mpc'.")

    if episodes == "all":
        start_indices = list(env.episode_start_indices)
    else:
        start_indices = list(env.episode_start_indices[: int(episodes)])

    print("=" * 72)
    print("IEEE69 Optimization Baseline Evaluation")
    print("=" * 72)
    print(f"topology     : {topology_case}")
    print(f"controller   : {controller}")
    print(f"mode         : {mode}")
    print(f"episodes     : {len(start_indices)}")
    print(f"horizon_t    : {horizon_t}")
    if controller == "smpc":
        print(f"n_scenarios  : {n_scenarios}")
    print(f"seed         : {seed}")
    print("=" * 72)

    all_rows = []

    for ep_id, start_idx in enumerate(start_indices):
        print()
        print("-" * 72)
        print(
            f"IEEE69 | "
            f"Topology {topology_case} | "
            f"Episode {ep_id + 1}/{len(start_indices)} | "
            f"start_index={start_idx}"
        )
        print("-" * 72)

        rows = evaluate_one_episode(
            controller=controller,
            env=env,
            scenario_data=scenario_data,
            point_data=point_data,
            episode_id=ep_id,
            start_index=start_idx,
            seed=seed + ep_id,
            horizon_t=horizon_t,
            n_scenarios=n_scenarios,
            topology_case=topology_case,
            print_every=print_every,
        )

        all_rows.extend(rows)

    step_df = pd.DataFrame(all_rows)
    episode_summary, aggregate_summary = summarize_results(step_df)

    if output_dir is None:
        base_dir = cfg.SMPC_RESULTS_DIR if controller == "smpc" else cfg.MPC_RESULTS_DIR

        if hasattr(cfg, "get_topology_output_dir"):
            output_dir = cfg.get_topology_output_dir(base_dir, topology_case)
        else:
            output_dir = Path(base_dir) / topology_case

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_path, episode_path, aggregate_path = _output_paths(
        controller=controller,
        output_dir=output_dir,
        horizon_t=horizon_t,
        n_scenarios=n_scenarios,
        topology_case=topology_case,
    )

    step_df.to_csv(step_path, index=False)
    episode_summary.to_csv(episode_path, index=False)
    aggregate_summary.to_csv(aggregate_path, index=False)

    print()
    print("=" * 72)
    print("Saved IEEE69 optimization baseline outputs")
    print("=" * 72)
    print("Step metrics      :", step_path)
    print("Episode summary   :", episode_path)
    print("Aggregate summary :", aggregate_path)
    print("=" * 72)

    if not aggregate_summary.empty:
        print()
        print(aggregate_summary.to_string(index=False))

    return {
        "step_df": step_df,
        "episode_summary": episode_summary,
        "aggregate_summary": aggregate_summary,
    }


if __name__ == "__main__":
    run_controller_evaluation(
        controller="smpc",
        episodes=1,
        horizon_t=cfg.HORIZON_T,
        n_scenarios=5,
        seed=42,
        mode="test",
        topology_case="TP1",
        print_every=24,
    )