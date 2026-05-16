from pathlib import Path
import argparse
import copy
import time

import pandas as pd
import torch

from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG


BEST_GNN_MODELS = [
    {
        "model_name": "gat_run1_step110000",
        "gnn_type": "gat",
        "checkpoint": "results/models/IEEE33/gat/ieee33_gat_run1_step110000_best.pt",
    },
    {
        "model_name": "gcn_run2_step165000",
        "gnn_type": "gcn",
        "checkpoint": "results/models/IEEE33/gcn/ieee33_gcn_run2_step165000_best.pt",
    },
    {
        "model_name": "tagconv_run0_step45000",
        "gnn_type": "tagconv",
        "checkpoint": "results/models/IEEE33/tagconv/ieee33_tagconv_run0_step45000_best.pt",
    },
]


def synchronize_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def build_agent(env, model_cfg, args, device):
    agent = GraphAwareMADDPG(
        node_feature_dim=int(env.observation_space["x"].shape[-1]),
        agent_obs_dim=int(env.agent_obs_dim),
        dess_buses=env.dess_buses,
        num_agents=env.num_agents,
        action_dim_per_agent=1,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        device=device,
        share_actor=args.share_actor,
        gnn_type=model_cfg["gnn_type"],
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_embedding_dim=args.gnn_embedding_dim,
        gnn_num_layers=args.gnn_num_layers,
    )

    agent.load(model_cfg["checkpoint"])
    agent.actor.eval()
    return agent


def reset_at_episode_start(env, start_index, seed):
    old_starts = env.episode_start_indices
    env.episode_start_indices = [int(start_index)]
    obs, info = env.reset(seed=seed)
    env.episode_start_indices = old_starts
    return obs, info


def measure_one_model(model_cfg, args, device):
    config = copy.deepcopy(IEEE33_CONFIG)
    env = DESSEnv(config=config, mode="test", seed=args.seed)

    agent = build_agent(env, model_cfg, args, device)

    start_indices = list(env.episode_start_indices)

    if args.episode_id >= len(start_indices):
        raise ValueError(
            f"episode_id={args.episode_id} is out of range. "
            f"Available episodes: 0 to {len(start_indices) - 1}"
        )

    start_index = start_indices[args.episode_id]

    obs, _ = reset_at_episode_start(
        env=env,
        start_index=start_index,
        seed=args.seed,
    )

    for _ in range(args.warmup):
        _ = agent.select_action(obs, noise_std=0.0)

    synchronize_if_cuda(device)

    rows = []
    done = False
    step = 0

    while not done:
        synchronize_if_cuda(device)
        t0 = time.perf_counter()

        action = agent.select_action(obs, noise_std=0.0)

        synchronize_if_cuda(device)
        t1 = time.perf_counter()

        inference_time_s = t1 - t0

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        rows.append(
            {
                "model_name": model_cfg["model_name"],
                "gnn_type": model_cfg["gnn_type"],
                "checkpoint": model_cfg["checkpoint"],
                "episode_id": args.episode_id,
                "start_index": int(start_index),
                "step": int(step),
                "inference_time_s": float(inference_time_s),
                "inference_time_ms": float(inference_time_s * 1000.0),
            }
        )

        obs = next_obs
        step += 1

    return pd.DataFrame(rows)


def summarize(times_df):
    return pd.DataFrame(
        [
            {
                "model_name": times_df["model_name"].iloc[0],
                "gnn_type": times_df["gnn_type"].iloc[0],
                "checkpoint": times_df["checkpoint"].iloc[0],
                "episode_id": int(times_df["episode_id"].iloc[0]),
                "steps": int(len(times_df)),
                "mean_inference_time_ms": float(times_df["inference_time_ms"].mean()),
                "std_inference_time_ms": float(times_df["inference_time_ms"].std()),
                "min_inference_time_ms": float(times_df["inference_time_ms"].min()),
                "max_inference_time_ms": float(times_df["inference_time_ms"].max()),
                "total_episode_inference_time_ms": float(
                    times_df["inference_time_ms"].sum()
                ),
                "control_interval_s": 900.0,
                "mean_time_fraction_of_15min_interval": float(
                    times_df["inference_time_s"].mean() / 900.0
                ),
            }
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="results/timing/IEEE33_best_gnn")

    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)

    parser.add_argument("--share_actor", action="store_true")
    parser.add_argument("--gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_embedding_dim", type=int, default=64)
    parser.add_argument("--gnn_num_layers", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_cfg in BEST_GNN_MODELS:
        print("========================================")
        print(f"Measuring: {model_cfg['model_name']}")
        print(f"GNN type:  {model_cfg['gnn_type']}")
        print(f"Checkpoint:{model_cfg['checkpoint']}")
        print("========================================")

        times_df = measure_one_model(model_cfg, args, device)
        summary_df = summarize(times_df)

        model_out_dir = out_base / model_cfg["model_name"]
        model_out_dir.mkdir(parents=True, exist_ok=True)

        times_path = model_out_dir / "inference_times_per_step.csv"
        summary_path = model_out_dir / "inference_time_summary.csv"

        times_df.to_csv(times_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        all_summaries.append(summary_df)

        print(summary_df.to_string(index=False))
        print(f"Saved per-step times to: {times_path}")
        print(f"Saved summary to:       {summary_path}")

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_path = out_base / "combined_gnn_inference_time_summary.csv"
    combined_summary.to_csv(combined_path, index=False)

    print("========================================")
    print("Combined GNN Timing Summary")
    print("========================================")
    print(combined_summary.to_string(index=False))
    print(f"Saved combined summary to: {combined_path}")


if __name__ == "__main__":
    main()