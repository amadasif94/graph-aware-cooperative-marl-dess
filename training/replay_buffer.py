"""
Graph replay buffer for graph-aware MADDPG DESS coordination.

Designed for DESSEnv observations:

    obs = {
        "x":          [num_buses, node_feature_dim],
        "edge_index": [2, num_edges],
        "edge_attr":  [num_edges, edge_feature_dim],
        "agent_obs": [num_agents, agent_obs_dim],
    }

The network topology edge_index is fixed and stored once.
Dynamic graph state x, edge_attr, and agent_obs are stored per transition.
"""

from typing import Dict, Optional, Tuple

import torch


class GraphReplayBuffer:
    """
    Replay buffer for graph-aware cooperative MADDPG.
    """

    def __init__(
        self,
        max_size: int,
        num_buses: int,
        node_feature_dim: int,
        num_agents: int,
        agent_obs_dim: int,
        action_dim_per_agent: int = 1,
        edge_index=None,
        edge_attr_shape: Optional[Tuple[int, ...]] = None,
        device: Optional[str] = None,
        store_on_device: bool = True,
    ):
        self.max_size = int(max_size)
        self.num_buses = int(num_buses)
        self.node_feature_dim = int(node_feature_dim)
        self.num_agents = int(num_agents)
        self.agent_obs_dim = int(agent_obs_dim)
        self.action_dim_per_agent = int(action_dim_per_agent)

        if self.max_size <= 0:
            raise ValueError("max_size must be positive.")
        if self.num_buses <= 0:
            raise ValueError("num_buses must be positive.")
        if self.node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be positive.")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        if self.agent_obs_dim <= 0:
            raise ValueError("agent_obs_dim must be positive.")
        if self.action_dim_per_agent <= 0:
            raise ValueError("action_dim_per_agent must be positive.")

        self.device = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.store_device = (
            self.device if store_on_device else torch.device("cpu")
        )

        self.ptr = 0
        self.size = 0
        self.full = False

        self.x = torch.empty(
            (self.max_size, self.num_buses, self.node_feature_dim),
            dtype=torch.float32,
            device=self.store_device,
        )

        self.next_x = torch.empty_like(self.x)

        self.agent_obs = torch.empty(
            (self.max_size, self.num_agents, self.agent_obs_dim),
            dtype=torch.float32,
            device=self.store_device,
        )

        self.next_agent_obs = torch.empty_like(self.agent_obs)

        self.actions = torch.empty(
            (
                self.max_size,
                self.num_agents,
                self.action_dim_per_agent,
            ),
            dtype=torch.float32,
            device=self.store_device,
        )

        self.rewards = torch.empty(
            (self.max_size, self.num_agents),
            dtype=torch.float32,
            device=self.store_device,
        )

        self.dones = torch.empty(
            (self.max_size, 1),
            dtype=torch.float32,
            device=self.store_device,
        )

        self.edge_index = None

        if edge_index is not None:
            self.edge_index = torch.as_tensor(
                edge_index,
                dtype=torch.long,
                device=self.store_device,
            )

        self.edge_attr = None
        self.next_edge_attr = None

        if edge_attr_shape is not None:
            self.edge_attr = torch.empty(
                (self.max_size, *edge_attr_shape),
                dtype=torch.float32,
                device=self.store_device,
            )

            self.next_edge_attr = torch.empty_like(self.edge_attr)

    def __len__(self):
        return self.size

    def _to_tensor(
        self,
        value,
        dtype=torch.float32,
    ) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().to(
                device=self.store_device,
                dtype=dtype,
            )

        return torch.as_tensor(
            value,
            dtype=dtype,
            device=self.store_device,
        )

    def _prepare_action(
        self,
        action,
    ) -> torch.Tensor:
        action = self._to_tensor(
            action,
            dtype=torch.float32,
        )

        if action.dim() == 1:
            action = action.unsqueeze(-1)

        expected_shape = (
            self.num_agents,
            self.action_dim_per_agent,
        )

        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"Expected action shape {expected_shape}, "
                f"got {tuple(action.shape)}"
            )

        return action

    def _prepare_reward(
        self,
        reward,
    ) -> torch.Tensor:
        reward = self._to_tensor(
            reward,
            dtype=torch.float32,
        )

        if reward.dim() == 0:
            reward = reward.repeat(self.num_agents)

        if reward.dim() == 2 and reward.shape[-1] == 1:
            reward = reward.squeeze(-1)

        expected_shape = (self.num_agents,)

        if tuple(reward.shape) != expected_shape:
            raise ValueError(
                f"Expected reward shape {expected_shape} or scalar, "
                f"got {tuple(reward.shape)}"
            )

        return reward

    def _prepare_done(
        self,
        done,
    ) -> torch.Tensor:
        done = self._to_tensor(
            done,
            dtype=torch.float32,
        )

        if done.dim() == 0:
            done = done.view(1)

        if done.numel() != 1:
            done = done.reshape(-1).any().float().view(1)

        return done.reshape(1)

    def add(
        self,
        obs: Dict,
        action,
        reward,
        next_obs: Dict,
        done,
    ):
        """
        Add one transition.
        """

        idx = self.ptr

        x = self._to_tensor(
            obs["x"],
            dtype=torch.float32,
        )

        next_x = self._to_tensor(
            next_obs["x"],
            dtype=torch.float32,
        )

        agent_obs = self._to_tensor(
            obs["agent_obs"],
            dtype=torch.float32,
        )

        next_agent_obs = self._to_tensor(
            next_obs["agent_obs"],
            dtype=torch.float32,
        )

        expected_x_shape = (
            self.num_buses,
            self.node_feature_dim,
        )

        expected_agent_obs_shape = (
            self.num_agents,
            self.agent_obs_dim,
        )

        if tuple(x.shape) != expected_x_shape:
            raise ValueError(
                f"Expected obs['x'] shape {expected_x_shape}, "
                f"got {tuple(x.shape)}"
            )

        if tuple(next_x.shape) != expected_x_shape:
            raise ValueError(
                f"Expected next_obs['x'] shape {expected_x_shape}, "
                f"got {tuple(next_x.shape)}"
            )

        if tuple(agent_obs.shape) != expected_agent_obs_shape:
            raise ValueError(
                f"Expected obs['agent_obs'] shape "
                f"{expected_agent_obs_shape}, got {tuple(agent_obs.shape)}"
            )

        if tuple(next_agent_obs.shape) != expected_agent_obs_shape:
            raise ValueError(
                f"Expected next_obs['agent_obs'] shape "
                f"{expected_agent_obs_shape}, "
                f"got {tuple(next_agent_obs.shape)}"
            )

        self.x[idx].copy_(x)
        self.next_x[idx].copy_(next_x)

        self.agent_obs[idx].copy_(agent_obs)
        self.next_agent_obs[idx].copy_(next_agent_obs)

        self.actions[idx].copy_(
            self._prepare_action(action)
        )

        self.rewards[idx].copy_(
            self._prepare_reward(reward)
        )

        self.dones[idx].copy_(
            self._prepare_done(done)
        )

        if self.edge_index is None and "edge_index" in obs:
            self.edge_index = torch.as_tensor(
                obs["edge_index"],
                dtype=torch.long,
                device=self.store_device,
            )

        if self.edge_attr is not None:
            if "edge_attr" not in obs:
                raise KeyError(
                    "edge_attr_shape was provided, but obs does not contain edge_attr."
                )

            edge_attr = self._to_tensor(
                obs["edge_attr"],
                dtype=torch.float32,
            )

            next_edge_attr_value = next_obs.get(
                "edge_attr",
                obs["edge_attr"],
            )

            next_edge_attr = self._to_tensor(
                next_edge_attr_value,
                dtype=torch.float32,
            )

            expected_edge_attr_shape = tuple(
                self.edge_attr.shape[1:]
            )

            if tuple(edge_attr.shape) != expected_edge_attr_shape:
                raise ValueError(
                    f"Expected edge_attr shape {expected_edge_attr_shape}, "
                    f"got {tuple(edge_attr.shape)}"
                )

            if tuple(next_edge_attr.shape) != expected_edge_attr_shape:
                raise ValueError(
                    f"Expected next_edge_attr shape "
                    f"{expected_edge_attr_shape}, "
                    f"got {tuple(next_edge_attr.shape)}"
                )

            self.edge_attr[idx].copy_(edge_attr)
            self.next_edge_attr[idx].copy_(next_edge_attr)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.full = self.size == self.max_size

    def sample(
        self,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch compatible with GraphAwareMADDPG.update().
        """

        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_size = int(batch_size)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        indices = torch.randint(
            low=0,
            high=self.size,
            size=(batch_size,),
            device=self.store_device,
        )

        batch = {
            "x": self.x[indices].to(self.device),
            "agent_obs": self.agent_obs[indices].to(self.device),
            "actions": self.actions[indices].to(self.device),
            "rewards": self.rewards[indices].to(self.device),
            "dones": self.dones[indices].to(self.device),
            "next_x": self.next_x[indices].to(self.device),
            "next_agent_obs": self.next_agent_obs[indices].to(self.device),
        }

        if self.edge_index is not None:
            batch["edge_index"] = self.edge_index.to(self.device)

        if self.edge_attr is not None:
            batch["edge_attr"] = self.edge_attr[indices].to(self.device)
            batch["next_edge_attr"] = self.next_edge_attr[indices].to(
                self.device
            )

        return batch

    def can_sample(
        self,
        batch_size: int,
    ) -> bool:
        return self.size >= int(batch_size)

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.full = False

    def save(
        self,
        path: str,
    ):
        payload = {
            "ptr": self.ptr,
            "size": self.size,
            "full": self.full,
            "max_size": self.max_size,
            "num_buses": self.num_buses,
            "node_feature_dim": self.node_feature_dim,
            "num_agents": self.num_agents,
            "agent_obs_dim": self.agent_obs_dim,
            "action_dim_per_agent": self.action_dim_per_agent,
            "x": self.x[: self.size].cpu(),
            "next_x": self.next_x[: self.size].cpu(),
            "agent_obs": self.agent_obs[: self.size].cpu(),
            "next_agent_obs": self.next_agent_obs[: self.size].cpu(),
            "actions": self.actions[: self.size].cpu(),
            "rewards": self.rewards[: self.size].cpu(),
            "dones": self.dones[: self.size].cpu(),
            "edge_index": None
            if self.edge_index is None
            else self.edge_index.cpu(),
            "edge_attr": None
            if self.edge_attr is None
            else self.edge_attr[: self.size].cpu(),
            "next_edge_attr": None
            if self.next_edge_attr is None
            else self.next_edge_attr[: self.size].cpu(),
        }

        torch.save(payload, path)

    def load(
        self,
        path: str,
    ):
        payload = torch.load(
            path,
            map_location=self.store_device,
        )

        loaded_size = int(payload["size"])

        if loaded_size > self.max_size:
            raise ValueError(
                f"Loaded buffer size {loaded_size} exceeds "
                f"current max_size {self.max_size}."
            )

        self.clear()

        self.x[:loaded_size].copy_(
            payload["x"].to(self.store_device)
        )

        self.next_x[:loaded_size].copy_(
            payload["next_x"].to(self.store_device)
        )

        self.agent_obs[:loaded_size].copy_(
            payload["agent_obs"].to(self.store_device)
        )

        self.next_agent_obs[:loaded_size].copy_(
            payload["next_agent_obs"].to(self.store_device)
        )

        self.actions[:loaded_size].copy_(
            payload["actions"].to(self.store_device)
        )

        self.rewards[:loaded_size].copy_(
            payload["rewards"].to(self.store_device)
        )

        self.dones[:loaded_size].copy_(
            payload["dones"].to(self.store_device)
        )

        if payload.get("edge_index", None) is not None:
            self.edge_index = payload["edge_index"].to(
                self.store_device
            )

        if payload.get("edge_attr", None) is not None:
            if self.edge_attr is None:
                edge_attr_shape = tuple(
                    payload["edge_attr"].shape[1:]
                )

                self.edge_attr = torch.empty(
                    (self.max_size, *edge_attr_shape),
                    dtype=torch.float32,
                    device=self.store_device,
                )

                self.next_edge_attr = torch.empty_like(
                    self.edge_attr
                )

            self.edge_attr[:loaded_size].copy_(
                payload["edge_attr"].to(self.store_device)
            )

            self.next_edge_attr[:loaded_size].copy_(
                payload["next_edge_attr"].to(self.store_device)
            )

        self.size = loaded_size
        self.ptr = loaded_size % self.max_size
        self.full = self.size == self.max_size


def build_replay_buffer_from_env_specs(
    env,
    max_size: int,
    device: Optional[str] = None,
    store_on_device: bool = True,
) -> GraphReplayBuffer:
    """
    Build replay buffer from environment attributes only.

    This intentionally does NOT call env.reset().
    """

    num_buses = int(env.num_buses)
    num_agents = int(env.num_agents)
    agent_obs_dim = int(env.agent_obs_dim)

    node_feature_dim = int(
        env.observation_space["x"].shape[-1]
    )

    edge_index = None
    edge_attr_shape = None

    if hasattr(env, "grid") and hasattr(env.grid, "edge_index"):
        edge_index = env.grid.edge_index

    if hasattr(env, "grid") and hasattr(env.grid, "edge_attr"):
        edge_attr_shape = tuple(env.grid.edge_attr.shape)

    return GraphReplayBuffer(
        max_size=max_size,
        num_buses=num_buses,
        node_feature_dim=node_feature_dim,
        num_agents=num_agents,
        agent_obs_dim=agent_obs_dim,
        action_dim_per_agent=1,
        edge_index=edge_index,
        edge_attr_shape=edge_attr_shape,
        device=device,
        store_on_device=store_on_device,
    )