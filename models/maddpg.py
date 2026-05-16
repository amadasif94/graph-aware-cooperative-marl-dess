"""
Graph-aware MADDPG trainer for cooperative DESS coordination.

Architecture:
    - Separate decentralized graph-aware actors
    - Agent-specific centralized graph-aware critics

For each DESS agent i:
    actor:  pi_i(o_i, h_i)
    critic: Q_i(h_i, h_global, O_t, A_t)

This follows standard MADDPG-style CTDE:
    - critics use joint information during centralized training
    - actors are used alone during decentralized execution
"""

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor import GraphAwareDESSActors
from models.critic import GraphAwareAgentCritic


class GraphAwareMADDPG:
    def __init__(
        self,
        node_feature_dim: int,
        agent_obs_dim: int,
        dess_buses,
        num_agents: int,
        action_dim_per_agent: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = 1.0,
        device: Optional[str] = None,
        share_actor: bool = False,
        gnn_type: str = "gcn",
        gnn_hidden_dim: int = 64,
        gnn_embedding_dim: int = 64,
        gnn_num_layers: int = 3,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
    ):
        self.device = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.grad_clip_norm = grad_clip_norm

        self.num_agents = int(num_agents)
        self.action_dim_per_agent = int(action_dim_per_agent)
        self.dess_buses = [int(b) for b in dess_buses]

        if len(self.dess_buses) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} DESS buses, "
                f"got {len(self.dess_buses)}."
            )

        self.actor = GraphAwareDESSActors(
            node_feature_dim=node_feature_dim,
            agent_obs_dim=agent_obs_dim,
            dess_buses=self.dess_buses,
            action_dim_per_agent=self.action_dim_per_agent,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_embedding_dim=gnn_embedding_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_type=gnn_type,
            actor_hidden_dims=actor_hidden_dims,
            share_actor=share_actor,
        ).to(self.device)

        self.critics = nn.ModuleList(
            [
                GraphAwareAgentCritic(
                    node_feature_dim=node_feature_dim,
                    agent_obs_dim=agent_obs_dim,
                    num_agents=self.num_agents,
                    agent_bus_idx=self.dess_buses[i],
                    action_dim_per_agent=self.action_dim_per_agent,
                    gnn_hidden_dim=gnn_hidden_dim,
                    gnn_embedding_dim=gnn_embedding_dim,
                    gnn_num_layers=gnn_num_layers,
                    gnn_type=gnn_type,
                    critic_hidden_dims=critic_hidden_dims,
                )
                for i in range(self.num_agents)
            ]
        ).to(self.device)

        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critics = deepcopy(self.critics).to(self.device)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
        )

        self.critic_optimizers = [
            torch.optim.AdamW(
                critic.parameters(),
                lr=critic_lr,
                weight_decay=weight_decay,
            )
            for critic in self.critics
        ]

        self.target_actor.eval()
        self.target_critics.eval()

    def _to_device(self, value):
        if value is None:
            return None

        if torch.is_tensor(value):
            return value.to(self.device)

        return torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, torch.Tensor],
        noise_std: float = 0.0,
    ):
        was_training = self.actor.training
        self.actor.eval()

        x = self._to_device(obs["x"])

        edge_index = torch.as_tensor(
            obs["edge_index"],
            dtype=torch.long,
            device=self.device,
        )

        edge_attr = self._to_device(obs.get("edge_attr", None))
        agent_obs = self._to_device(obs["agent_obs"])

        actions = self.actor.act(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            agent_obs=agent_obs,
            noise_std=noise_std,
        )

        if was_training:
            self.actor.train()

        return actions.detach().cpu().numpy()

    def _set_critics_requires_grad(self, requires_grad: bool):
        """
        Enable or disable critic gradients.

        During actor update, critics are used only to provide the
        policy-gradient signal. Their parameters should not accumulate
        gradients in that step.
        """

        for critic in self.critics:
            for param in critic.parameters():
                param.requires_grad = requires_grad

    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ):
        x = self._to_device(batch["x"])

        edge_index = torch.as_tensor(
            batch["edge_index"],
            dtype=torch.long,
            device=self.device,
        )

        edge_attr = self._to_device(batch.get("edge_attr", None))

        agent_obs = self._to_device(batch["agent_obs"])
        actions = self._to_device(batch["actions"])
        rewards = self._to_device(batch["rewards"])
        dones = self._to_device(batch["dones"])

        next_x = self._to_device(batch["next_x"])

        next_edge_index = torch.as_tensor(
            batch.get("next_edge_index", batch["edge_index"]),
            dtype=torch.long,
            device=self.device,
        )

        next_edge_attr = self._to_device(batch.get("next_edge_attr", None))

        if next_edge_attr is None:
            next_edge_attr = edge_attr

        next_agent_obs = self._to_device(batch["next_agent_obs"])

        if actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        expected_action_shape = (
            actions.shape[0],
            self.num_agents,
            self.action_dim_per_agent,
        )

        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(
                f"Expected actions shape {expected_action_shape}, "
                f"got {tuple(actions.shape)}."
            )

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)

        if rewards.dim() == 2 and rewards.shape[1] == 1:
            rewards = rewards.repeat(1, self.num_agents)

        if rewards.dim() != 2 or rewards.shape[1] != self.num_agents:
            raise ValueError(
                f"Expected rewards shape [B, {self.num_agents}], "
                f"got {tuple(rewards.shape)}."
            )

        if dones.dim() == 2:
            dones = dones.any(dim=1, keepdim=True).float()
        elif dones.dim() == 1:
            dones = dones.unsqueeze(-1).float()
        else:
            dones = dones.float()

        if dones.dim() != 2 or dones.shape[1] != 1:
            raise ValueError(
                f"Expected dones shape [B, 1], got {tuple(dones.shape)}."
            )

        with torch.no_grad():
            next_actions = self.target_actor(
                x=next_x,
                edge_index=next_edge_index,
                edge_attr=next_edge_attr,
                agent_obs=next_agent_obs,
            )

        critic_losses = []
        current_q_means = []
        target_q_means = []

        for agent_idx in range(self.num_agents):
            critic_i = self.critics[agent_idx]
            target_critic_i = self.target_critics[agent_idx]
            optimizer_i = self.critic_optimizers[agent_idx]

            reward_i = rewards[:, agent_idx:agent_idx + 1]

            with torch.no_grad():
                target_q_i = target_critic_i(
                    x=next_x,
                    edge_index=next_edge_index,
                    edge_attr=next_edge_attr,
                    agent_obs=next_agent_obs,
                    actions=next_actions,
                )

                q_target_i = (
                    reward_i
                    + self.gamma * (1.0 - dones) * target_q_i
                )

            current_q_i = critic_i(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                agent_obs=agent_obs,
                actions=actions,
            )

            critic_loss_i = F.mse_loss(
                current_q_i,
                q_target_i,
            )

            optimizer_i.zero_grad()
            critic_loss_i.backward()

            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    critic_i.parameters(),
                    self.grad_clip_norm,
                )

            optimizer_i.step()

            critic_losses.append(critic_loss_i.detach())
            current_q_means.append(current_q_i.detach().mean())
            target_q_means.append(q_target_i.detach().mean())

        self._set_critics_requires_grad(False)

        try:
            policy_actions = self.actor(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                agent_obs=agent_obs,
            )

            actor_loss = 0.0
            actor_q_means = []

            for agent_idx in range(self.num_agents):
                actor_q_i = self.critics[agent_idx](
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    agent_obs=agent_obs,
                    actions=policy_actions,
                )

                actor_loss = actor_loss - actor_q_i.mean()
                actor_q_means.append(actor_q_i.detach().mean())

            actor_loss = actor_loss / self.num_agents

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.grad_clip_norm,
                )

            self.actor_optimizer.step()

        finally:
            self._set_critics_requires_grad(True)

        self.soft_update(
            self.target_actor,
            self.actor,
        )

        self.soft_update(
            self.target_critics,
            self.critics,
        )

        critic_loss_tensor = torch.stack(critic_losses)
        current_q_tensor = torch.stack(current_q_means)
        target_q_tensor = torch.stack(target_q_means)
        actor_q_tensor = torch.stack(actor_q_means)

        logs = {
            "critic_loss": float(critic_loss_tensor.mean().item()),
            "actor_loss": float(actor_loss.item()),
            "mean_q": float(current_q_tensor.mean().item()),
            "mean_target_q": float(target_q_tensor.mean().item()),
            "mean_actor_q": float(actor_q_tensor.mean().item()),
        }

        for agent_idx in range(self.num_agents):
            logs[f"critic_loss_agent_{agent_idx}"] = float(
                critic_losses[agent_idx].item()
            )
            logs[f"mean_q_agent_{agent_idx}"] = float(
                current_q_means[agent_idx].item()
            )
            logs[f"mean_target_q_agent_{agent_idx}"] = float(
                target_q_means[agent_idx].item()
            )
            logs[f"mean_actor_q_agent_{agent_idx}"] = float(
                actor_q_means[agent_idx].item()
            )

        return logs

    def soft_update(
        self,
        target: nn.Module,
        source: nn.Module,
    ):
        with torch.no_grad():
            for target_param, source_param in zip(
                target.parameters(),
                source.parameters(),
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def save(
        self,
        path: str,
    ):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critics": self.target_critics.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizers": [
                    optimizer.state_dict()
                    for optimizer in self.critic_optimizers
                ],
                "dess_buses": self.dess_buses,
                "num_agents": self.num_agents,
                "action_dim_per_agent": self.action_dim_per_agent,
            },
            path,
        )

    def load(
        self,
        path: str,
    ):
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor.load_state_dict(checkpoint["actor"])
        self.critics.load_state_dict(checkpoint["critics"])

        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        if "critic_optimizers" in checkpoint:
            critic_optimizer_states = checkpoint["critic_optimizers"]

            if len(critic_optimizer_states) != len(self.critic_optimizers):
                raise ValueError(
                    "Checkpoint critic optimizer count does not match current model. "
                    f"Checkpoint has {len(critic_optimizer_states)}, "
                    f"model has {len(self.critic_optimizers)}."
                )

            for optimizer, state_dict in zip(
                self.critic_optimizers,
                critic_optimizer_states,
            ):
                optimizer.load_state_dict(state_dict)

        self.target_actor.eval()
        self.target_critics.eval()