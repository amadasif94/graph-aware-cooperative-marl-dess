"""
Training entry point for graph-aware MADDPG on IEEE 69-bus DESS environment.
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv
from training.trainer import GraphMADDPGTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(train: bool, seed: int):
    config = copy.deepcopy(IEEE69_CONFIG)
    mode = "train" if train else "val"

    return DESSEnv(
        config=config,
        mode=mode,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--noise_std", type=float, default=0.05)

    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gcn",
        choices=["gcn", "tagconv", "gat"],
    )

    parser.add_argument("--gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_embedding_dim", type=int, default=64)
    parser.add_argument("--gnn_num_layers", type=int, default=3)

    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=3)

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="results/checkpoints/maddpg_ieee69",
    )

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main():

    args = parse_args()

    set_seed(args.seed)

    device = (
        "cpu"
        if args.cpu
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_env = build_env(
        train=True,
        seed=args.seed,
    )

    eval_env = build_env(
        train=False,
        seed=args.seed + 1000,
    )

    train_config = {
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,

        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,

        "gamma": args.gamma,
        "tau": args.tau,

        "noise_std": args.noise_std,

        "gnn_type": args.gnn_type,
        "gnn_hidden_dim": args.gnn_hidden_dim,
        "gnn_embedding_dim": args.gnn_embedding_dim,
        "gnn_num_layers": args.gnn_num_layers,

        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "eval_episodes": args.eval_episodes,

        "checkpoint_dir": args.checkpoint_dir,

        "share_actor": False,
        "store_buffer_on_device": True,

        "action_dim_per_agent": 1,
    }

    Path(args.checkpoint_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("===================================")
    print("Graph-Aware MADDPG IEEE69 Training")
    print("===================================")

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    print(f"Total steps: {args.total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")

    print(f"Batch size: {args.batch_size}")

    print(f"GNN type: {args.gnn_type}")

    print(f"Checkpoint dir: {args.checkpoint_dir}")

    print("===================================")

    trainer = GraphMADDPGTrainer(
        env=train_env,
        eval_env=eval_env,
        config=train_config,
        device=device,
    )

    logs = trainer.train()

    print("Training complete.")
    print(f"Final checkpoint saved in: {args.checkpoint_dir}")
    print(f"Number of eval logs: {len(logs)}")


if __name__ == "__main__":
    main()