#!/usr/bin/env python3
from pathlib import Path
import argparse
import copy
import time

import pandas as pd
import torch

from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG


# ============================================================
# Best IEEE33 models actually saved in your folder:
#   results/models/IEEE33/gat/run_1.pt
#   results/models/IEEE33/gcn/run_2.pt
#   results/models/IEEE33/tagconv/run_0.pt
# ============================================================

BEST_GNN_MODELS = [
    {
        "model_name": "gat_run_1",
        "gnn_type": "gat",
        "checkpoint": "results/models/IEEE33/gat/run_1.pt",
    },
    {
        "model_name": "gcn_run_2",
        "gnn_type": "gcn",
        "checkpoint": "results/models/IEEE33/gcn/run_2.pt",
    },
    {
        "model_name": "tagconv_run_0",
        "gnn_type": "tagconv",
        "checkpoint": "results/models/IEEE33/tagconv/run_0.pt",
    },
]


def synchronize_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def resolve_checkpoint(model_cfg):
    checkpoint_path = Path(model_cfg["checkpoint"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def build_agent(env, model_cfg, checkpoint_path, args, device):
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

    agent.load(str(checkpoint_path))

    agent.actor.eval()

    if hasattr(agent, "critics"):
        agent.critics.eval()
    if hasattr(agent, "target_actor"):
        agent.target_actor.eval()
    if hasattr(agent, "target_critics"):
        agent.target_critics.eval()

    return agent


def reset_at_episode_start(env, start_index, seed):
    old_starts = env.episode_start_indices

    env.episode_start_indices = [int(start_index)]
    obs, info = env.reset(seed=seed)

    env.episode_start_indices = old_starts

    return obs, info


def measure_one_model(model_cfg, args, device):
    config = copy.deepcopy(IEEE33_CONFIG)

    env = DESSEnv(
        config=config,
        mode=args.mode,
        seed=args.seed,
    )

    checkpoint_path = resolve_checkpoint(model_cfg)

    agent = build_agent(
        env=env,
        model_cfg=model_cfg,
        checkpoint_path=checkpoint_path,
        args=args,
        device=device,
    )

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

    # Warmup forward passes
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = agent.select_action(obs, noise_std=0.0)

    synchronize_if_cuda(device)

    rows = []
    done = False
    step = 0

    with torch.inference_mode():
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
                    "checkpoint": str(checkpoint_path),
                    "device": str(device),
                    "mode": args.mode,
                    "episode_id": int(args.episode_id),
                    "start_index": int(start_index),
                    "step": int(step),
                    "inference_time_s": float(inference_time_s),
                    "inference_time_ms": float(inference_time_s * 1000.0),
                    "control_interval_s": float(args.control_interval_s),
                    "time_fraction_of_control_interval": float(
                        inference_time_s / args.control_interval_s
                    ),
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
                "device": times_df["device"].iloc[0],
                "mode": times_df["mode"].iloc[0],
                "episode_id": int(times_df["episode_id"].iloc[0]),
                "start_index": int(times_df["start_index"].iloc[0]),
                "steps": int(len(times_df)),

                "mean_inference_time_ms": float(
                    times_df["inference_time_ms"].mean()
                ),
                "std_inference_time_ms": float(
                    times_df["inference_time_ms"].std(ddof=1)
                ),
                "median_inference_time_ms": float(
                    times_df["inference_time_ms"].median()
                ),
                "p95_inference_time_ms": float(
                    times_df["inference_time_ms"].quantile(0.95)
                ),
                "min_inference_time_ms": float(
                    times_df["inference_time_ms"].min()
                ),
                "max_inference_time_ms": float(
                    times_df["inference_time_ms"].max()
                ),
                "total_episode_inference_time_ms": float(
                    times_df["inference_time_ms"].sum()
                ),

                "control_interval_s": float(
                    times_df["control_interval_s"].iloc[0]
                ),
                "mean_time_fraction_of_control_interval": float(
                    times_df["time_fraction_of_control_interval"].mean()
                ),
            }
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/timing/IEEE33_best_gnn",
    )

    parser.add_argument(
        "--control_interval_s",
        type=float,
        default=900.0,
        help="Control interval in seconds. Default 900 = 15 minutes.",
    )

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

    device = "cpu" if args.cpu else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("IEEE33 GNN computational-efficiency evaluation")
    print("=" * 72)
    print(f"Device             : {device}")
    print(f"Output dir         : {out_base}")
    print(f"Episode ID         : {args.episode_id}")
    print(f"Mode               : {args.mode}")
    print(f"Warmup passes      : {args.warmup}")
    print(f"Control interval s : {args.control_interval_s}")
    print("=" * 72)

    all_summaries = []

    for model_cfg in BEST_GNN_MODELS:
        print("\n" + "=" * 72)
        print(f"Measuring model : {model_cfg['model_name']}")
        print(f"GNN type        : {model_cfg['gnn_type']}")
        print(f"Checkpoint      : {model_cfg['checkpoint']}")
        print("=" * 72)

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

    print("\n" + "=" * 72)
    print("Combined GNN timing summary")
    print("=" * 72)
    print(combined_summary.to_string(index=False))
    print(f"Saved combined summary to: {combined_path}")


if __name__ == "__main__":
    main()