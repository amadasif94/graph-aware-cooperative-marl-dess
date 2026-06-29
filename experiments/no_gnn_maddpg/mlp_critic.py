"""
No-GNN agent-specific centralized critic for MADDPG-based DESS coordination.

This is the MLP / NN baseline critic for the GNN ablation study.

Graph-aware critic uses:
    Q_i(h_i, h_global, O_t, A_t)

This no-GNN critic uses:
    Q_i(O_t, A_t)

where:
    O_t = joint observations of all DESS agents
    A_t = joint actions of all DESS agents

Important:
    This critic is used ONLY during centralized training.

    Execution remains decentralized because the critic is not used during
    policy deployment.
"""

from typing import Optional, Sequence, Type

import torch
import torch.nn as nn


# ============================================================
# Generic MLP builder
# ============================================================

def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """
    Build a feedforward multilayer perceptron.
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


# ============================================================
# Agent-specific centralized no-GNN critic
# ============================================================

class MLPAgentCritic(nn.Module):
    """
    Agent-specific centralized critic without graph encoding.

    Critic input:
        [O_t, A_t]

    Critic output:
        Q_i(O_t, A_t)

    where:
        O_t = joint observations of all DESS agents
        A_t = joint actions of all DESS agents
    """

    def __init__(
        self,
        agent_obs_dim: int,
        num_agents: int,
        action_dim_per_agent: int = 1,
        critic_hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()

        self.agent_obs_dim = int(agent_obs_dim)
        self.num_agents = int(num_agents)
        self.action_dim_per_agent = int(action_dim_per_agent)

        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive.")

        if self.agent_obs_dim <= 0:
            raise ValueError("agent_obs_dim must be positive.")

        if self.action_dim_per_agent <= 0:
            raise ValueError("action_dim_per_agent must be positive.")

        critic_input_dim = (
            self.num_agents * self.agent_obs_dim
            + self.num_agents * self.action_dim_per_agent
        )

        self.q_network = build_mlp(
            input_dim=critic_input_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=nn.ReLU,
            output_activation=None,
        )

    # ============================================================
    # Input preparation
    # ============================================================

    def _prepare_inputs(
        self,
        agent_obs: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Normalize all inputs into batched format.

        Returns
        -------
        agent_obs:
            [B, A, D]

        actions:
            [B, A, action_dim]
        """

        if agent_obs.dim() == 2:
            agent_obs = agent_obs.unsqueeze(0)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0).unsqueeze(-1)

        elif actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        if agent_obs.dim() != 3:
            raise ValueError(
                "agent_obs must have shape [A, D] or [B, A, D], "
                f"got {tuple(agent_obs.shape)}"
            )

        if actions.dim() != 3:
            raise ValueError(
                "actions must have shape [A], [B, A], "
                "[A, action_dim], or [B, A, action_dim], "
                f"got {tuple(actions.shape)}"
            )

        batch_size = agent_obs.shape[0]

        expected_agent_obs_shape = (
            batch_size,
            self.num_agents,
            self.agent_obs_dim,
        )

        expected_action_shape = (
            batch_size,
            self.num_agents,
            self.action_dim_per_agent,
        )

        if tuple(agent_obs.shape) != expected_agent_obs_shape:
            raise ValueError(
                f"Expected agent_obs shape {expected_agent_obs_shape}, "
                f"got {tuple(agent_obs.shape)}"
            )

        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(
                f"Expected actions shape {expected_action_shape}, "
                f"got {tuple(actions.shape)}"
            )

        return agent_obs, actions

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(
        self,
        agent_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute agent-specific centralized Q-value.

        Parameters
        ----------
        agent_obs:
            [num_agents, agent_obs_dim]
            or [B, num_agents, agent_obs_dim]

        actions:
            [num_agents]
            or [B, num_agents]
            or [num_agents, action_dim]
            or [B, num_agents, action_dim]

        Returns
        -------
        torch.Tensor
            Q-value tensor with shape [B, 1].
        """

        agent_obs, actions = self._prepare_inputs(
            agent_obs=agent_obs,
            actions=actions,
        )

        flat_agent_obs = agent_obs.reshape(agent_obs.shape[0], -1)
        flat_actions = actions.reshape(actions.shape[0], -1)

        critic_input = torch.cat(
            [
                flat_agent_obs,
                flat_actions,
            ],
            dim=-1,
        )

        q_value = self.q_network(critic_input)

        return q_value