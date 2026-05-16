from pathlib import Path
import argparse
import copy
import time

import pandas as pd
import torch
import torch.nn as nn

from configs.ieee33_config import IEEE33_CONFIG
from environments.dess_env import DESSEnv


BEST_MLP_MODEL = {
    "model_name": "mlp_run2_step75000",
    "checkpoint": "results/models/IEEE33/mlp/ieee33_mlp_run2_step75000_best.pt",
}


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev_dim = int(input_dim)

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        prev_dim = int(hidden_dim)

    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    def __init__(self, agent_obs_dim, action_dim=1, hidden_dims=(256, 256)):
        super().__init__()

        self.agent_obs_dim = int(agent_obs_dim)
        self.action_dim = int(action_dim)

        self.net = build_mlp(
            input_dim=self.agent_obs_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
        )

    def forward(self, obs_i):
        return torch.tanh(self.net(obs_i))


class MLPDESSActors(nn.Module):
    def __init__(
        self,
        agent_obs_dim,
        num_agents,
        action_dim_per_agent=1,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        self.agent_obs_dim = int(agent_obs_dim)
        self.num_agents = int(num_agents)
        self.action_dim_per_agent = int(action_dim_per_agent)

        self.actors = nn.ModuleList(
            [
                MLPActor(
                    agent_obs_dim=self.agent_obs_dim,
                    action_dim=self.action_dim_per_agent,
                    hidden_dims=hidden_dims,
                )
                for _ in range(self.num_agents)
            ]
        )

    @torch.no_grad()
    def act(self, agent_obs):
        original_unbatched = agent_obs.dim() == 2

        if original_unbatched:
            agent_obs = agent_obs.unsqueeze(0)

        actions = []

        for i in range(self.num_agents):
            obs_i = agent_obs[:, i, :]
            action_i = self.actors[i](obs_i)
            actions.append(action_i)

        actions = torch.stack(actions, dim=1)

        if self.action_dim_per_agent == 1:
            actions = actions.squeeze(-1)

        if original_unbatched:
            actions = actions.squeeze(0)

        return torch.clamp(actions, -1.0, 1.0)


class MLPTimingAgent:
    def __init__(self, env, checkpoint_path, device):
        self.device = torch.device(device)

        self.actor = MLPDESSActors(
            agent_obs_dim=int(env.agent_obs_dim),
            num_agents=int(env.num_agents),
            action_dim_per_agent=1,
            hidden_dims=(256, 256),
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "actor" not in checkpoint:
            raise KeyError("Checkpoint does not contain key 'actor'.")

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()

    @torch.no_grad()
    def select_action(self, obs, noise_std=0.0):
        agent_obs = torch.as_tensor(
            obs["agent_obs"],
            dtype=torch.float32,
            device=self.device,
        )

        actions = self.actor.act(agent_obs)

        if noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * float(noise_std)
            actions = torch.clamp(actions, -1.0, 1.0)

        return actions.detach().cpu().numpy()


def reset_at_episode_start(env, start_index, seed):
    old_starts = env.episode_start_indices
    env.episode_start_indices = [int(start_index)]
    obs, info = env.reset(seed=seed)
    env.episode_start_indices = old_starts
    return obs, info


def synchronize_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def measure_one_episode(env, agent, args, device):
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
                "model_name": BEST_MLP_MODEL["model_name"],
                "gnn_type": "mlp_no_gnn",
                "checkpoint": BEST_MLP_MODEL["checkpoint"],
                "episode_id": int(args.episode_id),
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
                "gnn_type": "mlp_no_gnn",
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
    parser.add_argument("--out_dir", type=str, default="results/timing/IEEE33_best_mlp")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    config = copy.deepcopy(IEEE33_CONFIG)
    env = DESSEnv(config=config, mode="test", seed=args.seed)

    agent = MLPTimingAgent(
        env=env,
        checkpoint_path=BEST_MLP_MODEL["checkpoint"],
        device=device,
    )

    times_df = measure_one_episode(env, agent, args, device)
    summary_df = summarize(times_df)

    out_dir = Path(args.out_dir) / BEST_MLP_MODEL["model_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    times_path = out_dir / "inference_times_per_step.csv"
    summary_path = out_dir / "inference_time_summary.csv"

    times_df.to_csv(times_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    combined_path = Path(args.out_dir) / "combined_mlp_inference_time_summary.csv"
    summary_df.to_csv(combined_path, index=False)

    print("========================================")
    print("MLP Inference Timing Summary")
    print("========================================")
    print(summary_df.to_string(index=False))
    print("========================================")
    print(f"Saved per-step times to: {times_path}")
    print(f"Saved summary to:       {summary_path}")
    print(f"Saved combined summary: {combined_path}")


if __name__ == "__main__":
    main()