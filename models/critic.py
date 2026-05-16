"""
Graph-aware agent-specific centralized critic for MADDPG-based
distributed energy storage system (DESS) coordination.

This module implements an agent-specific centralized critic:

    Q_i(h_i, h_global, O_t, A_t)

where:

    h_i       : topology-aware embedding of agent i's DESS bus
    h_global  : global graph embedding of the entire feeder
    O_t       : joint observations of all agents
    A_t       : joint actions of all agents

Important:
    This critic is used ONLY during centralized training.

    Execution remains decentralized because the critic is not used
    during policy deployment.

Architecture:
    1. A graph neural network encodes the feeder topology.
    2. Node embeddings are generated for every bus.
    3. The embedding corresponding to the agent's DESS bus is extracted.
    4. A global feeder embedding is computed using graph pooling.
    5. The critic receives:
            [h_i, h_global, O_t, A_t]
       and outputs:
            Q_i
"""

from typing import Optional, Sequence, Type

import torch
import torch.nn as nn

from models.gnn_encoder import GNNEncoder


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
# Agent-specific centralized critic
# ============================================================

class GraphAwareAgentCritic(nn.Module):
    """
    Agent-specific graph-aware centralized critic.

    Critic input:
        [h_i, h_global, O_t, A_t]

    Critic output:
        Q_i(h_i, h_global, O_t, A_t)

    where:
        h_i      = local DESS-bus embedding for agent i
        h_global = global feeder embedding
    """

    def __init__(
        self,
        node_feature_dim: int,
        agent_obs_dim: int,
        num_agents: int,
        agent_bus_idx: int,
        action_dim_per_agent: int = 1,
        gnn_hidden_dim: int = 64,
        gnn_embedding_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_type: str = "gcn",
        tag_k: int = 2,
        gat_heads: int = 4,
        gnn_dropout: float = 0.0,
        critic_hidden_dims: Sequence[int] = (256, 256),
    ):
        """
        Initialize agent-specific centralized critic.

        Parameters
        ----------
        node_feature_dim : int
            Node feature dimension.

        agent_obs_dim : int
            Observation dimension for one agent.

        num_agents : int
            Total number of MARL agents.

        agent_bus_idx : int
            Bus index corresponding to this critic's DESS agent.

        action_dim_per_agent : int
            Action dimension per agent.

        gnn_hidden_dim : int
            Hidden GNN dimension.

        gnn_embedding_dim : int
            Output graph embedding dimension.

        gnn_num_layers : int
            Number of GNN layers.

        gnn_type : str
            GNN layer type.

        critic_hidden_dims : Sequence[int]
            Hidden dimensions of critic MLP.
        """

        super().__init__()

        self.node_feature_dim = int(node_feature_dim)
        self.agent_obs_dim = int(agent_obs_dim)

        self.num_agents = int(num_agents)
        self.agent_bus_idx = int(agent_bus_idx)

        self.action_dim_per_agent = int(action_dim_per_agent)

        self.gnn_embedding_dim = int(gnn_embedding_dim)

        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive.")

        # =====================================================
        # Graph neural network encoder
        # =====================================================

        self.gnn_encoder = GNNEncoder(
            input_dim=self.node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            embedding_dim=self.gnn_embedding_dim,
            num_layers=gnn_num_layers,
            gnn_type=gnn_type,
            tag_k=tag_k,
            gat_heads=gat_heads,
            dropout=gnn_dropout,
            use_layer_norm=True,
        )

        # =====================================================
        # Critic MLP
        # =====================================================

        critic_input_dim = (
            self.gnn_embedding_dim                  # h_i
            + self.gnn_embedding_dim               # h_global
            + self.num_agents * self.agent_obs_dim
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
        x: torch.Tensor,
        agent_obs: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Normalize all inputs into batched format.

        Returns:
            x           : [B, N, F]
            agent_obs   : [B, A, D]
            actions     : [B, A, action_dim]
        """

        if x.dim() == 2:
            x = x.unsqueeze(0)

        if agent_obs.dim() == 2:
            agent_obs = agent_obs.unsqueeze(0)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0).unsqueeze(-1)

        elif actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        if x.dim() != 3:
            raise ValueError(
                "x must have shape [N, F] or [B, N, F], "
                f"got {tuple(x.shape)}"
            )

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

        batch_size = x.shape[0]

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
                f"Expected agent_obs shape "
                f"{expected_agent_obs_shape}, "
                f"got {tuple(agent_obs.shape)}"
            )

        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(
                f"Expected actions shape "
                f"{expected_action_shape}, "
                f"got {tuple(actions.shape)}"
            )

        return x, agent_obs, actions

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        agent_obs: torch.Tensor,
        actions: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute agent-specific centralized Q-value.

        Returns
        -------
        torch.Tensor
            Q-value tensor with shape [B, 1].
        """

        # ----------------------------------------------------
        # Normalize tensor shapes
        # ----------------------------------------------------

        x, agent_obs, actions = self._prepare_inputs(
            x=x,
            agent_obs=agent_obs,
            actions=actions,
        )

        # ----------------------------------------------------
        # Compute topology-aware node embeddings
        # ----------------------------------------------------

        node_embeddings = self.gnn_encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # ----------------------------------------------------
        # Extract local DESS-bus embedding h_i
        # ----------------------------------------------------

        agent_embedding = node_embeddings[:, self.agent_bus_idx, :]

        # ----------------------------------------------------
        # Compute global feeder embedding h_global
        # ----------------------------------------------------

        global_graph_embedding = node_embeddings.mean(dim=1)

        # ----------------------------------------------------
        # Flatten multi-agent observations/actions
        # ----------------------------------------------------

        flat_agent_obs = agent_obs.reshape(agent_obs.shape[0], -1)

        flat_actions = actions.reshape(actions.shape[0], -1)

        # ----------------------------------------------------
        # Build critic input vector
        # ----------------------------------------------------

        critic_input = torch.cat(
            [
                agent_embedding,
                global_graph_embedding,
                flat_agent_obs,
                flat_actions,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Predict agent-specific Q-value
        # ----------------------------------------------------

        q_value = self.q_network(critic_input)

        return q_value