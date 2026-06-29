"""
Graph-aware decentralized actor networks for MADDPG-based DESS coordination.

Design:
    Environment returns:
        x, edge_index, edge_attr, agent_obs

    GNN encoder computes:
        H_t = GNNEncoder(x, edge_index, edge_attr)

    For each DESS agent i:
        h_i(t) = H_t[dess_bus_i]
        actor_input_i = concat(agent_obs_i, h_i(t))
        a_i(t) = actor_i(actor_input_i)

Execution is decentralized because each actor receives only:
    - its own local agent observation
    - the GNN embedding of its own DESS bus
"""

from typing import Iterable, Optional, Sequence, Type

import torch
import torch.nn as nn

from models.gnn_encoder import GNNEncoder


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


class DESSActor(nn.Module):
    """
    Decentralized actor for one DESS agent.

    Input:
        concat(agent_obs_i, h_i)

    Output:
        normalized DESS action in [-1, 1]
    """

    def __init__(
        self,
        agent_obs_dim: int,
        embedding_dim: int,
        action_dim: int = 1,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()

        self.agent_obs_dim = int(agent_obs_dim)
        self.embedding_dim = int(embedding_dim)
        self.action_dim = int(action_dim)

        self.net = build_mlp(
            input_dim=self.agent_obs_dim + self.embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            activation=nn.ReLU,
            output_activation=None,
        )

    def forward(
        self,
        agent_obs_i: torch.Tensor,
        embedding_i: torch.Tensor,
    ) -> torch.Tensor:
        original_unbatched = agent_obs_i.dim() == 1

        if original_unbatched:
            agent_obs_i = agent_obs_i.unsqueeze(0)
            embedding_i = embedding_i.unsqueeze(0)

        if agent_obs_i.dim() != 2:
            raise ValueError(
                "agent_obs_i must have shape [D] or [B, D], "
                f"got {tuple(agent_obs_i.shape)}"
            )

        if embedding_i.dim() != 2:
            raise ValueError(
                "embedding_i must have shape [E] or [B, E], "
                f"got {tuple(embedding_i.shape)}"
            )

        if agent_obs_i.shape[-1] != self.agent_obs_dim:
            raise ValueError(
                f"Expected agent_obs_dim={self.agent_obs_dim}, "
                f"got {agent_obs_i.shape[-1]}"
            )

        if embedding_i.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {embedding_i.shape[-1]}"
            )

        actor_input = torch.cat([agent_obs_i, embedding_i], dim=-1)
        action = torch.tanh(self.net(actor_input))

        if original_unbatched:
            action = action.squeeze(0)

        return action


class GraphAwareDESSActors(nn.Module):
    """
    Graph-aware decentralized actor module for all DESS agents.

    The environment provides:
        x, edge_index, edge_attr, agent_obs

    The actor module computes:
        H_t = GNNEncoder(x, edge_index, edge_attr)

    Then for each DESS agent i:
        h_i(t) = H_t[dess_bus_i]
        a_i(t) = actor_i(concat(agent_obs_i, h_i(t)))

    If share_actor=True, all DESS agents share one actor network.
    If share_actor=False, each DESS agent has its own actor network.
    """

    def __init__(
        self,
        node_feature_dim: int,
        agent_obs_dim: int,
        dess_buses: Iterable[int],
        action_dim_per_agent: int = 1,
        gnn_hidden_dim: int = 64,
        gnn_embedding_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_type: str = "gcn",
        tag_k: int = 2,
        gat_heads: int = 4,
        gnn_dropout: float = 0.0,
        actor_hidden_dims: Sequence[int] = (256, 256),
        share_actor: bool = True,
    ):
        super().__init__()

        self.node_feature_dim = int(node_feature_dim)
        self.agent_obs_dim = int(agent_obs_dim)
        self.action_dim_per_agent = int(action_dim_per_agent)
        self.gnn_embedding_dim = int(gnn_embedding_dim)
        self.share_actor = bool(share_actor)

        dess_bus_tensor = torch.tensor(list(dess_buses), dtype=torch.long)

        if dess_bus_tensor.numel() == 0:
            raise ValueError("dess_buses cannot be empty.")

        self.register_buffer("dess_buses", dess_bus_tensor)
        self.num_agents = int(dess_bus_tensor.numel())

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

        if self.share_actor:
            self.actor = DESSActor(
                agent_obs_dim=self.agent_obs_dim,
                embedding_dim=self.gnn_embedding_dim,
                action_dim=self.action_dim_per_agent,
                hidden_dims=actor_hidden_dims,
            )
        else:
            self.actors = nn.ModuleList(
                [
                    DESSActor(
                        agent_obs_dim=self.agent_obs_dim,
                        embedding_dim=self.gnn_embedding_dim,
                        action_dim=self.action_dim_per_agent,
                        hidden_dims=actor_hidden_dims,
                    )
                    for _ in range(self.num_agents)
                ]
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        agent_obs: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute actions for all DESS agents.

        Inputs:
            x:
                [num_buses, node_feature_dim]
                or [B, num_buses, node_feature_dim]

            edge_index:
                [2, num_edges]

            agent_obs:
                [num_agents, agent_obs_dim]
                or [B, num_agents, agent_obs_dim]

            edge_attr:
                optional edge features

        Returns:
            actions:
                [num_agents] or [B, num_agents] when action_dim_per_agent = 1
        """

        original_unbatched = x.dim() == 2

        if original_unbatched:
            x = x.unsqueeze(0)
            agent_obs = agent_obs.unsqueeze(0)

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

        batch_size = x.shape[0]

        if agent_obs.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch: x batch={batch_size}, "
                f"agent_obs batch={agent_obs.shape[0]}"
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

        node_embeddings = self.gnn_encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        dess_embeddings = node_embeddings[:, self.dess_buses, :]

        action_list = []

        for agent_idx in range(self.num_agents):
            obs_i = agent_obs[:, agent_idx, :]
            emb_i = dess_embeddings[:, agent_idx, :]

            if self.share_actor:
                action_i = self.actor(obs_i, emb_i)
            else:
                action_i = self.actors[agent_idx](obs_i, emb_i)

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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        agent_obs: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        Select actions for rollout or evaluation.

        Adds Gaussian exploration noise if noise_std > 0.
        """

        actions = self.forward(
            x=x,
            edge_index=edge_index,
            agent_obs=agent_obs,
            edge_attr=edge_attr,
        )

        if noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * float(noise_std)

        return torch.clamp(actions, -1.0, 1.0)