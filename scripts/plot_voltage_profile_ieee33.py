import argparse
import copy
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG
from experiments.topology_generalization.topology_utils import build_topology_config


def build_env(topology_case, seed):
    if topology_case == "TP1":
        config = copy.deepcopy(IEEE33_CONFIG)
    else:
        config = build_topology_config(topology_case)

    return DESSEnv(config=config, mode="test", seed=seed)


def reset_at_episode_start(env, start_index, seed):
    old_starts = env.episode_start_indices
    env.episode_start_indices = [int(start_index)]
    obs, info = env.reset(seed=seed)
    env.episode_start_indices = old_starts
    return obs, info


def build_agent(env, args, device):
    agent = GraphAwareMADDPG(
        node_feature_dim=int(env.observation_space["x"].shape[-1]),
        agent_obs_dim=int(env.agent_obs_dim),
        dess_buses=env.dess_buses,
        num_agents=env.num_agents,
        action_dim_per_agent=1,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        device=device,
        share_actor=False,
        gnn_type=args.gnn_type,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_embedding_dim=args.gnn_embedding_dim,
        gnn_num_layers=args.gnn_num_layers,
    )

    agent.load(args.checkpoint)
    agent.actor.eval()
    agent.critics.eval()
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gnn_type", default="gcn", choices=["gcn", "gat", "tag"])
    parser.add_argument("--topology_case", default="TP6")
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="ieee33_tp6_voltage_profile.png")

    parser.add_argument("--gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_embedding_dim", type=int, default=64)
    parser.add_argument("--gnn_num_layers", type=int, default=2)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = build_env(args.topology_case, args.seed)
    agent = build_agent(env, args, device)

    obs, _ = reset_at_episode_start(env, args.episode_start, args.seed)

    voltage_history = []
    min_voltage_history = []

    done = False

    while not done:
        action = agent.select_action(obs, noise_std=0.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        voltage = np.asarray(next_obs["x"][:, 5], dtype=float)

        voltage_history.append(voltage)
        min_voltage_history.append(float(np.min(voltage)))

        obs = next_obs

    t_star = int(np.argmin(min_voltage_history))
    v_star = voltage_history[t_star]

    bus = np.arange(1, len(v_star) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(bus, v_star, marker="o", linewidth=1.5, label=f"{args.gnn_type.upper()} - {args.topology_case}")
    plt.axhline(0.95, linestyle="--", linewidth=1, label="Voltage limits")
    plt.axhline(1.05, linestyle="--", linewidth=1)
    plt.xlabel("Bus index")
    plt.ylabel("Voltage magnitude (p.u.)")
    plt.title(f"IEEE33 {args.topology_case} voltage profile at worst timestep")
    plt.legend()
    plt.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)

    print(f"Saved: {args.out}")
    print(f"Worst timestep: {t_star}")
    print(f"Minimum voltage: {min_voltage_history[t_star]:.4f} p.u.")


if __name__ == "__main__":
    main()