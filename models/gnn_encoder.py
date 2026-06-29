"""
GNN encoder for graph-aware cooperative MARL DESS coordination.

This module converts graph-structured distribution-network observations into
topology-aware node embeddings.

Input:
    x           Node feature tensor
                shape: [num_buses, node_feature_dim]
                or     [batch_size, num_buses, node_feature_dim]

    edge_index  Directed graph connectivity
                shape: [2, num_edges]

    edge_attr   Optional edge attributes
                shape: [num_edges, edge_feature_dim]
                Currently accepted for interface compatibility, but not used
                by GCNConv, TAGConv, or GATv2Conv.

Output:
    node_embeddings
                shape: [num_buses, embedding_dim]
                or     [batch_size, num_buses, embedding_dim]

The encoder also provides a helper function for extracting only the DESS-bus
embeddings used by decentralized DESS agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, TAGConv, GATv2Conv
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required for models/gnn_encoder.py. "
        "Install PyTorch Geometric."
    ) from exc


class GNNEncoder(nn.Module):
    """
    Topology-aware graph neural network encoder.

    Supported layer types:
        - 'gcn'      : Graph Convolutional Network
        - 'tagconv'  : Topology Adaptive Graph Convolution
        - 'gat'      : Graph Attention Network using GATv2Conv
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        embedding_dim=64,
        num_layers=3,
        gnn_type="gcn",
        tag_k=2,
        gat_heads=4,
        dropout=0.0,
        use_layer_norm=True,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.gnn_type = str(gnn_type).lower()
        self.tag_k = int(tag_k)
        self.gat_heads = int(gat_heads)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)

        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        if self.gnn_type not in ["gcn", "tagconv", "gat"]:
            raise ValueError(
                "Unsupported gnn_type '{}'. Use one of: 'gcn', 'tagconv', 'gat'.".format(
                    self.gnn_type
                )
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = self._build_layer_dims()

        for layer_idx in range(self.num_layers):
            in_channels = dims[layer_idx]
            out_channels = dims[layer_idx + 1]

            conv = self._make_conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                is_last_layer=(layer_idx == self.num_layers - 1),
            )

            self.convs.append(conv)

            if self.use_layer_norm:
                self.norms.append(nn.LayerNorm(out_channels))
            else:
                self.norms.append(nn.Identity())

    def _build_layer_dims(self):
        """
        Build channel dimensions for all GNN layers.
        """

        if self.num_layers == 1:
            return [self.input_dim, self.embedding_dim]

        dims = [self.input_dim]

        for _ in range(self.num_layers - 1):
            dims.append(self.hidden_dim)

        dims.append(self.embedding_dim)

        return dims

    def _make_conv_layer(self, in_channels, out_channels, is_last_layer=False):
        """
        Construct one graph convolution layer.
        """

        if self.gnn_type == "gcn":
            return GCNConv(
                in_channels=in_channels,
                out_channels=out_channels,
            )

        if self.gnn_type == "tagconv":
            return TAGConv(
                in_channels=in_channels,
                out_channels=out_channels,
                K=self.tag_k,
            )

        if self.gnn_type == "gat":
            if is_last_layer:
                return GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=1,
                    concat=False,
                )

            if out_channels % self.gat_heads != 0:
                raise ValueError(
                    "For GAT hidden layers, out_channels must be divisible by gat_heads. "
                    "Got out_channels={} and gat_heads={}.".format(
                        out_channels,
                        self.gat_heads,
                    )
                )

            return GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels // self.gat_heads,
                heads=self.gat_heads,
                concat=True,
            )

        raise RuntimeError("Invalid gnn_type reached unexpectedly.")

    def _repeat_edge_index_for_batch(self, edge_index, batch_size, num_nodes, device):
        """
        Repeat a single graph's edge_index for batched graph processing.

        PyTorch Geometric expects all graphs in a batch to be represented as one
        disconnected large graph. This function offsets node indices for each
        batch element.
        """

        edge_index = edge_index.to(device=device, dtype=torch.long)

        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges], got {}.".format(
                    tuple(edge_index.shape)
                )
            )

        num_edges = edge_index.shape[1]

        offsets = (
            torch.arange(batch_size, device=device, dtype=torch.long)
            .view(batch_size, 1, 1)
            * num_nodes
        )

        edge_index_batched = edge_index.view(1, 2, num_edges) + offsets
        edge_index_batched = edge_index_batched.permute(1, 0, 2).reshape(
            2,
            batch_size * num_edges,
        )

        return edge_index_batched

    def forward(self, x, edge_index, edge_attr=None):
        """
        Compute topology-aware node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor with shape [N, F] or [B, N, F].

        edge_index : torch.Tensor
            Graph connectivity with shape [2, E].

        edge_attr : torch.Tensor or None
            Optional edge attributes. Present for interface compatibility.
            Not used by the current GCN/TAGConv/GATv2Conv layers.

        Returns
        -------
        torch.Tensor
            Node embeddings with shape [N, embedding_dim] or [B, N, embedding_dim].
        """

        del edge_attr  # Kept only for future edge-aware extensions.

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        original_is_unbatched = x.dim() == 2

        if original_is_unbatched:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                "x must have shape [N, F] or [B, N, F], got {}.".format(
                    tuple(x.shape)
                )
            )

        batch_size, num_nodes, feature_dim = x.shape

        if feature_dim != self.input_dim:
            raise ValueError(
                "Expected input feature dimension {}, got {}.".format(
                    self.input_dim,
                    feature_dim,
                )
            )

        device = x.device
        edge_index_batched = self._repeat_edge_index_for_batch(
            edge_index=edge_index,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
        )

        h = x.reshape(batch_size * num_nodes, feature_dim)

        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index_batched)
            h = self.norms[layer_idx](h)

            if layer_idx < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        h = h.reshape(batch_size, num_nodes, self.embedding_dim)

        if original_is_unbatched:
            h = h.squeeze(0)

        return h

    def extract_agent_embeddings(self, node_embeddings, dess_buses):
        """
        Extract embeddings corresponding to DESS buses.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Shape [N, embedding_dim] or [B, N, embedding_dim].

        dess_buses : list, tuple, or torch.Tensor
            Zero-based DESS bus indices.

        Returns
        -------
        torch.Tensor
            DESS embeddings with shape [num_agents, embedding_dim]
            or [B, num_agents, embedding_dim].
        """

        if not torch.is_tensor(node_embeddings):
            node_embeddings = torch.as_tensor(node_embeddings, dtype=torch.float32)

        dess_buses = torch.as_tensor(
            dess_buses,
            dtype=torch.long,
            device=node_embeddings.device,
        )

        if node_embeddings.dim() == 2:
            return node_embeddings[dess_buses, :]

        if node_embeddings.dim() == 3:
            return node_embeddings[:, dess_buses, :]

        raise ValueError(
            "node_embeddings must have shape [N, D] or [B, N, D], got {}.".format(
                tuple(node_embeddings.shape)
            )
        )


def build_gnn_encoder_from_config(config):
    """
    Convenience constructor using the project configuration dictionary.
    """

    graph_cfg = config["graph"]

    input_dim = int(graph_cfg["node_feature_dim"])

    gnn_cfg = config.get("gnn", {})

    hidden_dim = int(gnn_cfg.get("hidden_dim", 64))
    embedding_dim = int(gnn_cfg.get("embedding_dim", 64))
    num_layers = int(gnn_cfg.get("num_layers", 3))
    gnn_type = str(gnn_cfg.get("type", "gcn"))
    tag_k = int(gnn_cfg.get("tag_k", 2))
    gat_heads = int(gnn_cfg.get("gat_heads", 4))
    dropout = float(gnn_cfg.get("dropout", 0.0))
    use_layer_norm = bool(gnn_cfg.get("use_layer_norm", True))

    return GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        tag_k=tag_k,
        gat_heads=gat_heads,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    )