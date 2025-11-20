import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNEncoder(nn.Module):
    """
    Robust GNN encoder supporting GCN and multi-head GAT with:
      - per-layer residual projection (no shape-mismatch)
      - node-type Embedding fused with numeric node features
      - mean+max pooling to produce graph embeddings

    Expected `data` in forward():
      - data.x         -> FloatTensor shape (num_nodes, input_dim)
      - data.node_ids  -> LongTensor shape (num_nodes,)  (integer ids into node_vocab)
      - data.edge_index-> LongTensor shape (2, num_edges)
      - data.batch     -> LongTensor shape (num_nodes,) mapping nodes -> graph idx
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_layers: int = 3,
        gnn_type: str = 'GAT',       
        heads: int = 4,               
        dropout: float = 0.1,
        node_vocab_size: int = 1000
    ):
        super().__init__()

        # store hyperparams
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.heads = heads
        self.dropout = dropout

        # Embedding for discrete node types (node_ids)
        self.node_embedding = nn.Embedding(node_vocab_size, hidden_dim)

        # Project numeric input features into hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Containers for GNN layers, batch norms, and residual projections
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # per-layer projection to match conv output dim

        # Build layers sequentially, tracking the current input dimension (prev_dim)
        prev_dim = hidden_dim  # after input_projection + node_embedding fuse

        for i in range(num_layers):
            # Decide behavior for this layer
            is_last = (i == num_layers - 1)

            if self.gnn_type == 'GCN':
                # GCN: keep out_dim = hidden_dim (stable dims across layers)
                out_dim = hidden_dim
                conv = GCNConv(prev_dim, out_dim)

            else:  # 'GAT'
                # For intermediate GAT layers we often use concat=True to increase capacity
                layer_heads = self.heads if not is_last else 1
                concat = (not is_last)  # concat intermediate heads, average/merge on last
                out_channels = hidden_dim  # per-head output size

                # GATConv(in_channels, out_channels, heads=..., concat=...)
                conv = GATConv(prev_dim, out_channels, heads=layer_heads, dropout=dropout, concat=concat)

                # compute the final output dimension of this conv
                out_dim = out_channels * layer_heads if concat else out_channels

            # register conv and matching BatchNorm
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

            # create a linear that maps prev_dim -> out_dim for residual addition
            self.res_projs.append(nn.Linear(prev_dim, out_dim))

            # update prev_dim for next layer
            prev_dim = out_dim

        # after loop prev_dim is the node feature dim after the last conv
        final_node_dim = prev_dim

        # final projection (graph-level): we concatenate mean+max -> 2 * final_node_dim
        self.output_projection = nn.Sequential(
            nn.Linear(final_node_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

        logger.info(
            f"GNNEncoder(init) type={gnn_type}, layers={num_layers}, hidden_dim={hidden_dim}, "
            f"heads={heads}, final_node_dim={final_node_dim}, output_dim={output_dim}"
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass producing graph-level embeddings.

        Args:
            data: torch_geometric Batch with attributes
                  x: (num_nodes, input_dim)
                  node_ids: (num_nodes,) long
                  edge_index: (2, num_edges)
                  batch: (num_nodes,) long

        Returns:
            graph_embedding: Tensor (batch_size, output_dim)
        """
        # Basic unpack
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Validate presence of node_ids
        if not hasattr(data, 'node_ids'):
            raise AttributeError("data must contain 'node_ids' LongTensor (num_nodes,) to use node_embedding")

        # 1) project numeric features and fetch node-type embeddings
        x_num = self.input_projection(x)            # (N, hidden_dim)
        x_type = self.node_embedding(data.node_ids) # (N, hidden_dim)

        # fuse numeric + type embeddings (elementwise sum)
        x = x_num + x_type                          # (N, hidden_dim) initial node features

        # 2) apply GNN layers with matched residual projections
        for i, (conv, bn, proj) in enumerate(zip(self.convs, self.batch_norms, self.res_projs)):
            # message passing
            x_new = conv(x, edge_index)            # shape depends on conv (see layer construction)
            # batch norm expects shape (N, num_features)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # residual projection maps previous x to x_new's feature dimension
            res = proj(x)
            x = res + x_new

        # 3) global pooling (per-graph)
        mean_pool = global_mean_pool(x, batch) # (batch_size, final_node_dim)
        max_pool = global_max_pool(x, batch)   # (batch_size, final_node_dim)

        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)  # (batch_size, 2*final_node_dim)

        # 4) final projection to output_dim
        graph_embedding = self.output_projection(graph_embedding)  # (batch_size, output_dim)

        return graph_embedding

    def get_node_embeddings(self, data: Batch) -> torch.Tensor:
        """
        Return node-level embeddings after the final GNN layer.
        Args:
            data: Batch with same expected fields as forward()
        Returns:
            x: Tensor (num_nodes, final_node_dim)
        """
        if not hasattr(data, 'node_ids'):
            raise AttributeError("data must contain 'node_ids' LongTensor (num_nodes,) to use node_embedding")

        x, edge_index = data.x, data.edge_index

        x = self.input_projection(x)
        x = x + self.node_embedding(data.node_ids)

        # same per-layer residual behaviour as forward
        for conv, bn, proj in zip(self.convs, self.batch_norms, self.res_projs):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = proj(x) + x_new

        return x
