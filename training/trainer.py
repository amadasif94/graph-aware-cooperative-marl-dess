"""
Trainer for graph-aware MADDPG DESS coordination.

Training loop:
    warm-up random rollout
    -> store graph transitions
    -> update MADDPG
    -> collect policy rollout
    -> periodically evaluate
    -> save checkpoint
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from models.maddpg import GraphAwareMADDPG
from training.replay_buffer import (
    GraphReplayBuffer,
    build_replay_buffer_from_env_specs,
)


class GraphMADDPGTrainer:
    def __init__(
        self,
        env,
        eval_env=None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        self.env = env
        self.eval_env = eval_env
        self.config = {} if config is None else config

        self.device = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.num_buses = int(env.num_buses)
        self.num_agents = int(env.num_agents)
        self.agent_obs_dim = int(env.agent_obs_dim)
        self.node_feature_dim = int(env.observation_space["x"].shape[-1])
        self.action_dim_per_agent = int(self.config.get("action_dim_per_agent", 1))

        self.batch_size = int(self.config.get("batch_size", 64))
        self.buffer_size = int(self.config.get("buffer_size", 100000))
        self.warmup_steps = int(self.config.get("warmup_steps", 5000))
        self.total_steps = int(self.config.get("total_steps", 100000))
        self.update_after = int(self.config.get("update_after", self.warmup_steps))
        self.update_every = int(self.config.get("update_every", 1))
        self.eval_every = int(self.config.get("eval_every", 5000))
        self.save_every = int(self.config.get("save_every", 10000))
        self.noise_std = float(self.config.get("noise_std", 0.05))
        self.eval_episodes = int(self.config.get("eval_episodes", 3))
        self.max_eval_steps = int(
            self.config.get("max_eval_steps", getattr(env, "episode_length", 96))
        )
        self.verbose = bool(self.config.get("verbose", True))

        self.checkpoint_dir = Path(
            self.config.get("checkpoint_dir", "results/checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.replay_buffer: GraphReplayBuffer = build_replay_buffer_from_env_specs(
            env=self.env,
            max_size=self.buffer_size,
            device=str(self.device),
            store_on_device=bool(self.config.get("store_buffer_on_device", True)),
        )

        self.agent = GraphAwareMADDPG(
            node_feature_dim=self.node_feature_dim,
            agent_obs_dim=self.agent_obs_dim,
            dess_buses=self.env.dess_buses,
            num_agents=self.num_agents,
            action_dim_per_agent=self.action_dim_per_agent,
            gamma=float(self.config.get("gamma", 0.99)),
            tau=float(self.config.get("tau", 0.005)),
            actor_lr=float(self.config.get("actor_lr", 1e-4)),
            critic_lr=float(self.config.get("critic_lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
            grad_clip_norm=self.config.get("grad_clip_norm", 1.0),
            device=str(self.device),
            share_actor=bool(self.config.get("share_actor", False)),
            gnn_type=str(self.config.get("gnn_type", "gcn")),
            gnn_hidden_dim=int(self.config.get("gnn_hidden_dim", 64)),
            gnn_embedding_dim=int(self.config.get("gnn_embedding_dim", 64)),
            gnn_num_layers=int(self.config.get("gnn_num_layers", 3)),
            actor_hidden_dims=tuple(
                self.config.get("actor_hidden_dims", (256, 256))
            ),
            critic_hidden_dims=tuple(
                self.config.get("critic_hidden_dims", (256, 256))
            ),
        )

        self.global_step = 0
        self.episode_count = 0

    def _reset_env(self, env):
        output = env.reset()

        if isinstance(output, tuple):
            obs, info = output
        else:
            obs, info = output, {}

        return obs, info

    def _step_env(self, env, action):
        output = env.step(action)

        if len(output) == 5:
            next_obs, reward, terminated, truncated, info = output
            done = bool(terminated or truncated)
        elif len(output) == 4:
            next_obs, reward, done, info = output
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError("env.step(action) must return either 4 or 5 values.")

        return next_obs, reward, done, terminated, truncated, info

    def _random_action(self):
        return np.random.uniform(
            low=-0.75,                #Amad
            high=0.75,
            size=(self.num_agents,),
        ).astype(np.float32)

    def collect_step(self, obs: Dict, random_action: bool = False):
        if random_action:
            requested_action = self._random_action()
        else:
            requested_action = self.agent.select_action(
                obs,
                noise_std=self.noise_std,
            )

        next_obs, reward, done, terminated, truncated, info = self._step_env(
            self.env,
            requested_action,
        )

        accepted_action = info.get("accepted_action", requested_action)

        self.replay_buffer.add(
            obs=obs,
            action=accepted_action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )

        self.global_step += 1

        if done:
            self.episode_count += 1
            next_obs, _ = self._reset_env(self.env)

        return next_obs, reward, done, info

    def warmup(self):
        obs, _ = self._reset_env(self.env)

        while len(self.replay_buffer) < self.warmup_steps:
            obs, _, _, _ = self.collect_step(
                obs,
                random_action=True,
            )

            if self.verbose and len(self.replay_buffer) % 1000 == 0:
                print(
                    f"warmup buffer={len(self.replay_buffer)} "
                    f"global_step={self.global_step}"
                )

        return obs

    def train(self):
        obs = self.warmup()
        logs = []

        if self.verbose:
            print(
                f"Starting training from global_step={self.global_step}, "
                f"buffer_size={len(self.replay_buffer)}"
            )

        while self.global_step < self.total_steps:
            obs, reward, done, info = self.collect_step(
                obs,
                random_action=False,
            )

            update_log = None

            if (
                self.global_step >= self.update_after
                and self.replay_buffer.can_sample(self.batch_size)
                and self.global_step % self.update_every == 0
            ):
                batch = self.replay_buffer.sample(self.batch_size)
                update_log = self.agent.update(batch)

            if self.eval_env is not None and self.global_step % self.eval_every == 0:
                eval_log = self.evaluate()

                logs.append(
                    {
                        "step": self.global_step,
                        "episode": self.episode_count,
                        "update": update_log,
                        "eval": eval_log,
                    }
                )

                critic_loss_str = (
                    "None"
                    if update_log is None
                    else f"{update_log['critic_loss']:.6f}"
                )
                actor_loss_str = (
                    "None"
                    if update_log is None
                    else f"{update_log['actor_loss']:.6f}"
                )

                print(
                    f"step={self.global_step} "
                    f"episode={self.episode_count} "
                    f"eval_reward={eval_log['mean_reward']:.4f} "
                    f"eval_len={eval_log['mean_length']:.1f} "
                    f"critic_loss={critic_loss_str} "
                    f"actor_loss={actor_loss_str}"
                )

            if self.global_step % self.save_every == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"maddpg_step_{self.global_step}.pt"
                )

        self.save_checkpoint(self.checkpoint_dir / "maddpg_final.pt")

        return logs

    @torch.no_grad()
    def evaluate(self):
        if self.eval_env is None:
            raise ValueError(
                "eval_env must be provided for evaluation. "
                "Do not evaluate on the training environment."
            )

        env = self.eval_env

        episode_rewards = []
        episode_lengths = []
        episode_feasible = []

        for ep in range(self.eval_episodes):
            obs, _ = self._reset_env(env)

            done = False
            ep_reward = 0.0
            ep_len = 0
            all_feasible = True

            while not done and ep_len < self.max_eval_steps:
                action = self.agent.select_action(
                    obs,
                    noise_std=0.0,
                )

                obs, reward, done, _, _, info = self._step_env(
                    env,
                    action,
                )

                reward_arr = np.asarray(reward, dtype=np.float32)
                ep_reward += float(np.mean(reward_arr))
                ep_len += 1

                if not bool(info.get("feasible", True)):
                    all_feasible = False

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            episode_feasible.append(all_feasible)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "all_feasible_rate": float(np.mean(episode_feasible)),
        }

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))

    def load_checkpoint(self, path):
        self.agent.load(str(path))