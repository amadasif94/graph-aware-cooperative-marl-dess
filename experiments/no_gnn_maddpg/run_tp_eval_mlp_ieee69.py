"""
Evaluate trained no-GNN / MLP MADDPG policy under IEEE 69-bus topology
reconfiguration benchmark cases TP1-TP9.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from environments.dess_env import DESSEnv
from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG

from experiments.topology_generalization.tp_configs_ieee69 import (
    list_topology_cases,
)

from experiments.topology_generalization.topology_utils_ieee69 import (
    build_topology_config,
)


MLP_POLICY_NAME = "mlp_maddpg"


def build_agent(env, args, device):
    agent = MLPMADDPG(
        agent_obs_dim=int(env.agent_obs_dim),
        num_agents=env.num_agents,
        action_dim_per_agent=1,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        device=device,
        share_actor=args.share_actor,
    )

    agent.load(args.checkpoint)

    agent.actor.eval()
    agent.critics.eval()
    agent.target_actor.eval()
    agent.target_critics.eval()

    return agent


def reset_at_episode_start(env, start_index, seed):
    old_starts = env.episode_start_indices
    env.episode_start_indices = [int(start_index)]

    obs, info = env.reset(seed=seed)

    env.episode_start_indices = old_starts

    return obs, info


def get_action(policy, agent, env, obs):
    if policy in ["maddpg", "mlp", MLP_POLICY_NAME]:
        return agent.select_action(obs, noise_std=0.0)

    if policy == "zero":
        return np.zeros(env.num_agents, dtype=np.float32)

    if policy == "random":
        return env.action_space.sample()

    raise ValueError(f"Unknown policy: {policy}")


def evaluate_one_episode(
    env,
    agent,
    policy,
    topology_case,
    episode_id,
    start_index,
    seed,
):
    obs, _ = reset_at_episode_start(
        env=env,
        start_index=start_index,
        seed=seed,
    )

    rows = []
    done = False
    step = 0

    while not done:
        requested_action = get_action(
            policy=policy,
            agent=agent,
            env=env,
            obs=obs,
        )

        next_obs, reward, terminated, truncated, info = env.step(
            requested_action
        )

        done = bool(terminated or truncated)

        reward = np.asarray(reward, dtype=np.float64)
        kpis = info["kpis"]

        accepted_action = np.asarray(
            info["accepted_action"],
            dtype=np.float64,
        )
        requested_action = np.asarray(
            info["requested_action"],
            dtype=np.float64,
        )

        dess_power_kw_full = np.asarray(
            info["dess_power_kw"],
            dtype=np.float64,
        )
        dess_power_agents = dess_power_kw_full[env.dess_buses]

        grid_import_kw = float(kpis["grid_import_kw"])
        curtailment_kw = float(kpis["curtailment_kw"])
        price = float(kpis["price"])

        grid_import_mwh = (
            max(0.0, grid_import_kw)
            * env.delta_t_hours
            / 1000.0
        )

        curtailment_mwh = (
            curtailment_kw
            * env.delta_t_hours
            / 1000.0
        )

        throughput_mwh = (
            np.sum(np.abs(dess_power_agents))
            * env.delta_t_hours
            / 1000.0
        )

        energy_cost = price * grid_import_mwh

        soc_values = [
            float(battery.get_soc())
            for battery in env.batteries
        ]

        row = {
            "topology_case": topology_case,
            "policy": policy,
            "episode_id": int(episode_id),
            "step": int(step),
            "date_time": info["date_time"],
            "start_index": int(start_index),

            "reward_mean": float(np.mean(reward)),
            "reward_sum": float(np.sum(reward)),

            "grid_import_kw": float(grid_import_kw),
            "grid_import_mwh": float(grid_import_mwh),
            "energy_cost": float(energy_cost),

            "curtailment_kw": float(curtailment_kw),
            "curtailment_mwh": float(curtailment_mwh),

            "voltage_deviation": float(kpis["voltage_deviation"]),
            "grid_stress": float(kpis["grid_stress"]),

            "min_voltage_pu": float(info["min_voltage_pu"]),
            "max_voltage_pu": float(info["max_voltage_pu"]),
            "max_line_current_pu": float(info["max_line_current_pu"]),

            "max_voltage_violation": float(info["max_voltage_violation"]),
            "max_line_current_violation": float(
                info["max_line_current_violation"]
            ),

            "feasible": bool(info["feasible"]),
            "converged": bool(info["converged"]),
            "infeasible_action": bool(info["infeasible_action"]),

            "throughput_mwh": float(throughput_mwh),
        }

        for i in range(env.num_agents):
            row[f"requested_action_agent_{i}"] = float(requested_action[i])
            row[f"accepted_action_agent_{i}"] = float(accepted_action[i])
            row[f"dess_power_kw_agent_{i}"] = float(dess_power_agents[i])
            row[f"soc_agent_{i}"] = float(soc_values[i])

        rows.append(row)

        obs = next_obs
        step += 1

    return rows


def summarize(step_df):
    episode_summary = step_df.groupby(
        [
            "topology_case",
            "policy",
            "episode_id",
            "start_index",
        ]
    ).agg(
        total_reward_mean=("reward_mean", "sum"),
        total_reward_team=("reward_sum", "sum"),

        total_energy_cost=("energy_cost", "sum"),
        total_grid_import_mwh=("grid_import_mwh", "sum"),
        total_curtailment_mwh=("curtailment_mwh", "sum"),
        total_throughput_mwh=("throughput_mwh", "sum"),

        mean_voltage_deviation=("voltage_deviation", "mean"),
        max_voltage_deviation=("voltage_deviation", "max"),

        min_voltage_pu=("min_voltage_pu", "min"),
        max_voltage_pu=("max_voltage_pu", "max"),
        max_line_current_pu=("max_line_current_pu", "max"),

        max_voltage_violation=("max_voltage_violation", "max"),
        max_line_current_violation=("max_line_current_violation", "max"),

        infeasible_requested_count=("infeasible_action", "sum"),
        feasible_rate=("feasible", "mean"),
        converged_rate=("converged", "mean"),
    ).reset_index()

    aggregate_summary = episode_summary.groupby(
        [
            "topology_case",
            "policy",
        ]
    ).agg(
        episodes=("episode_id", "count"),

        mean_total_reward_mean=("total_reward_mean", "mean"),
        std_total_reward_mean=("total_reward_mean", "std"),

        mean_total_reward_team=("total_reward_team", "mean"),
        std_total_reward_team=("total_reward_team", "std"),

        mean_energy_cost=("total_energy_cost", "mean"),
        mean_grid_import_mwh=("total_grid_import_mwh", "mean"),
        mean_curtailment_mwh=("total_curtailment_mwh", "mean"),
        mean_throughput_mwh=("total_throughput_mwh", "mean"),

        mean_voltage_deviation=("mean_voltage_deviation", "mean"),

        worst_min_voltage_pu=("min_voltage_pu", "min"),
        worst_max_voltage_pu=("max_voltage_pu", "max"),
        worst_line_current_pu=("max_line_current_pu", "max"),

        worst_voltage_violation=("max_voltage_violation", "max"),
        worst_line_current_violation=("max_line_current_violation", "max"),

        mean_infeasible_requested_count=(
            "infeasible_requested_count",
            "mean",
        ),
        mean_feasible_rate=("feasible_rate", "mean"),
        mean_converged_rate=("converged_rate", "mean"),
    ).reset_index()

    return episode_summary, aggregate_summary


def make_plots(step_df, episode_summary, figure_dir):
    del step_df

    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "total_reward_mean",
        "total_energy_cost",
        "total_grid_import_mwh",
        "total_curtailment_mwh",
        "total_throughput_mwh",
        "mean_voltage_deviation",
        "min_voltage_pu",
        "max_voltage_pu",
        "max_line_current_pu",
        "feasible_rate",
        "infeasible_requested_count",
    ]

    for metric in metrics:
        plt.figure(figsize=(12, 5))

        labels = []
        data = []

        grouped = episode_summary.groupby(
            [
                "topology_case",
                "policy",
            ]
        )

        for (tp, policy), group in grouped:
            data.append(group[metric].values)
            labels.append(f"{tp}-{policy}")

        if len(data) == 0:
            plt.close()
            continue

        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"IEEE69 {metric}")
        plt.tight_layout()

        plt.savefig(
            figure_dir / f"{metric}_by_topology_policy.png",
            dpi=300,
        )

        plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
    )

    parser.add_argument("--episodes", type=str, default="all")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--run_name",
        type=str,
        default="ieee69_tp_generalization_mlp_maddpg_eval",
    )

    parser.add_argument(
        "--topologies",
        nargs="+",
        default=None,
        help="Topology cases to evaluate, e.g. TP1 TP2 TP3 TP8 TP9.",
    )

    parser.add_argument(
        "--csv_dir",
        type=str,
        default="results/topology_generalization/csv_ieee69",
    )

    parser.add_argument(
        "--figure_dir",
        type=str,
        default="results/topology_generalization/figures_ieee69",
    )

    parser.add_argument("--include_zero", action="store_true")
    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)

    parser.add_argument("--share_actor", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        "cpu"
        if args.cpu
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    if args.topologies is None:
        topology_cases = list_topology_cases(include_baseline=True)
    else:
        topology_cases = [tp.upper() for tp in args.topologies]

    policies = [MLP_POLICY_NAME]

    if args.include_zero:
        policies.append("zero")

    if args.include_random:
        policies.append("random")

    csv_dir = Path(args.csv_dir) / args.run_name
    figure_dir = Path(args.figure_dir) / args.run_name

    csv_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================")
    print("No-GNN / MLP MADDPG IEEE69 Topology Generalization Eval")
    print("====================================================")
    print("Device:", device)
    print("Mode:", args.mode)
    print("Checkpoint:", args.checkpoint)
    print("Topologies:", topology_cases)
    print("Policies:", policies)
    print("share_actor:", args.share_actor)
    print("CSV output:", csv_dir)
    print("Figure output:", figure_dir)
    print("====================================================")

    all_rows = []

    for tp in topology_cases:
        print()
        print("====================================================")
        print("Evaluating IEEE69 topology:", tp)
        print("====================================================")

        tp_config = build_topology_config(
            case_name=tp,
            validate=True,
        )

        probe_env = DESSEnv(
            config=tp_config,
            mode=args.mode,
            seed=args.seed,
        )

        if args.episodes == "all":
            start_indices = list(probe_env.episode_start_indices)
        else:
            num_episodes = int(args.episodes)
            start_indices = list(
                probe_env.episode_start_indices[:num_episodes]
            )

        print("Description:", tp_config["topology_case"]["description"])
        print("Open edges:", tp_config["topology_case"]["open_edges"])
        print("Close edges:", tp_config["topology_case"]["close_edges"])
        print("Episodes:", len(start_indices))
        print("In-service lines:", len(probe_env.grid.in_service_line_df))
        print("edge_index shape:", probe_env.grid.edge_index.shape)

        for policy in policies:
            env = DESSEnv(
                config=tp_config,
                mode=args.mode,
                seed=args.seed,
            )

            agent = None

            if policy == MLP_POLICY_NAME:
                agent = build_agent(
                    env=env,
                    args=args,
                    device=device,
                )

            for ep_id, start_idx in enumerate(start_indices):
                rows = evaluate_one_episode(
                    env=env,
                    agent=agent,
                    policy=policy,
                    topology_case=tp,
                    episode_id=ep_id,
                    start_index=start_idx,
                    seed=args.seed + ep_id,
                )

                all_rows.extend(rows)

                ep_reward_mean = sum(
                    row["reward_mean"]
                    for row in rows
                )
                ep_reward_team = sum(
                    row["reward_sum"]
                    for row in rows
                )
                ep_cost = sum(
                    row["energy_cost"]
                    for row in rows
                )
                ep_feasible_rate = np.mean(
                    [
                        row["feasible"]
                        for row in rows
                    ]
                )
                ep_infeasible_requested = sum(
                    row["infeasible_action"]
                    for row in rows
                )

                print(
                    f"topology={tp} "
                    f"policy={policy} "
                    f"episode={ep_id} "
                    f"start_index={start_idx} "
                    f"reward_mean={ep_reward_mean:.4f} "
                    f"reward_team={ep_reward_team:.4f} "
                    f"cost={ep_cost:.4f} "
                    f"feasible_rate={ep_feasible_rate:.3f} "
                    f"infeasible_requested={ep_infeasible_requested}"
                )

    step_df = pd.DataFrame(all_rows)

    if step_df.empty:
        raise RuntimeError("Evaluation produced no rows.")

    episode_summary, aggregate_summary = summarize(step_df)

    step_path = csv_dir / "step_metrics.csv"
    episode_path = csv_dir / "episode_summary.csv"
    aggregate_path = csv_dir / "aggregate_summary.csv"

    step_df.to_csv(step_path, index=False)
    episode_summary.to_csv(episode_path, index=False)
    aggregate_summary.to_csv(aggregate_path, index=False)

    make_plots(
        step_df=step_df,
        episode_summary=episode_summary,
        figure_dir=figure_dir,
    )

    print()
    print("IEEE69 MLP topology generalization evaluation complete.")
    print("Saved CSV files:")
    print(step_path)
    print(episode_path)
    print(aggregate_path)
    print("Saved figures:")
    print(figure_dir)


if __name__ == "__main__":
    main()