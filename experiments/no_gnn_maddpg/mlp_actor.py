"""
No-GNN decentralized actor networks for MADDPG-based DESS coordination.

This is the MLP / NN baseline for the GNN ablation study.

Design:
    Environment still returns:
        x, edge_index, edge_attr, agent_obs

    But this actor ignores:
        x, edge_index, edge_attr

    For each DESS agent i:
        actor_input_i = agent_obs_i
        a_i(t) = actor_i(agent_obs_i)

Execution is decentralized because each actor receives only its own
local agent observation.
"""

from typing import Optional, Sequence, Type

import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """
    Build a feedforward MLP.
    """

    layers = []
    prev_dim = int(input_dim)

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(activation())
        prev_dim = int(hidden_dim)

    layers.append(nn.Linear(prev_dim, int(output_dim)))

    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


class MLPDESSActor(nn.Module):
    """
    Decentralized no-GNN actor for one DESS agent.

    Input:
        agent_obs_i

    Output:
        normalized DESS action in [-1, 1]
    """

    def __init__(
        self,
        agent_obs_dim: int,
        action_dim: int = 1,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()

        self.agent_obs_dim = int(agent_obs_dim)
        self.action_dim = int(action_dim)

        self.net = build_mlp(
            input_dim=self.agent_obs_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            activation=nn.ReLU,
            output_activation=None,
        )

    def forward(self, agent_obs_i: torch.Tensor) -> torch.Tensor:
        original_unbatched = agent_obs_i.dim() == 1

        if original_unbatched:
            agent_obs_i = agent_obs_i.unsqueeze(0)

        if agent_obs_i.dim() != 2:
            raise ValueError(
                "agent_obs_i must have shape [D] or [B, D], "
                f"got {tuple(agent_obs_i.shape)}"
            )

        if agent_obs_i.shape[-1] != self.agent_obs_dim:
            raise ValueError(
                f"Expected agent_obs_dim={self.agent_obs_dim}, "
                f"got {agent_obs_i.shape[-1]}"
            )

        action = torch.tanh(self.net(agent_obs_i))

        if original_unbatched:
            action = action.squeeze(0)

        return action


class MultiAgentMLPActors(nn.Module):
    """
    No-GNN decentralized actor module for all DESS agents.

    If share_actor=True:
        all DESS agents share one MLP actor.

    If share_actor=False:
        each DESS agent has its own MLP actor.

    This module intentionally ignores graph inputs so it can be used as the
    no-GNN / NN baseline against GCN, TAGConv, and GAT.
    """

    def __init__(
        self,
        agent_obs_dim: int,
        num_agents: int,
        action_dim_per_agent: int = 1,
        actor_hidden_dims: Sequence[int] = (256, 256),
        share_actor: bool = True,
    ):
        super().__init__()

        self.agent_obs_dim = int(agent_obs_dim)
        self.num_agents = int(num_agents)
        self.action_dim_per_agent = int(action_dim_per_agent)
        self.share_actor = bool(share_actor)

        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive.")

        if self.action_dim_per_agent <= 0:
            raise ValueError("action_dim_per_agent must be positive.")

        if self.share_actor:
            self.actor = MLPDESSActor(
                agent_obs_dim=self.agent_obs_dim,
                action_dim=self.action_dim_per_agent,
                hidden_dims=actor_hidden_dims,
            )
        else:
            self.actors = nn.ModuleList(
                [
                    MLPDESSActor(
                        agent_obs_dim=self.agent_obs_dim,
                        action_dim=self.action_dim_per_agent,
                        hidden_dims=actor_hidden_dims,
                    )
                    for _ in range(self.num_agents)
                ]
            )

    def forward(
        self,
        agent_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute actions for all DESS agents.

        Parameters
        ----------
        agent_obs:
            [num_agents, agent_obs_dim]
            or [B, num_agents, agent_obs_dim]

        Returns
        -------
        actions:
            [num_agents] or [B, num_agents] when action_dim_per_agent = 1
            otherwise [num_agents, action_dim] or [B, num_agents, action_dim]
        """

        original_unbatched = agent_obs.dim() == 2

        if original_unbatched:
            agent_obs = agent_obs.unsqueeze(0)

        if agent_obs.dim() != 3:
            raise ValueError(
                "agent_obs must have shape [A, D] or [B, A, D], "
                f"got {tuple(agent_obs.shape)}"
            )

        if agent_obs.shape[1] != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} agents, got {agent_obs.shape[1]}"
            )

        if agent_obs.shape[2] != self.agent_obs_dim:
            raise ValueError(
                f"Expected agent_obs_dim={self.agent_obs_dim}, "
                f"got {agent_obs.shape[2]}"
            )

        action_list = []

        for agent_idx in range(self.num_agents):
            obs_i = agent_obs[:, agent_idx, :]

            if self.share_actor:
                action_i = self.actor(obs_i)
            else:
                action_i = self.actors[agent_idx](obs_i)

            action_list.append(action_i)

        actions = torch.stack(action_list, dim=1)

        if self.action_dim_per_agent == 1:
            actions = actions.squeeze(-1)

        if original_unbatched:
            actions = actions.squeeze(0)

        return actions

    @torch.no_grad()
    def act(
        self,
        agent_obs: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        Select actions for rollout or evaluation.

        Adds Gaussian exploration noise if noise_std > 0.
        """

        actions = self.forward(agent_obs=agent_obs)

        if noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * float(noise_std)

        return torch.clamp(actions, -1.0, 1.0)