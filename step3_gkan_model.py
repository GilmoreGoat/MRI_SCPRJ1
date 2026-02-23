import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from step2_kan_layer import KANLinear

class GKANConv(MessagePassing):
    """
    Custom Graph Convolution Layer that replaces the standard linear transformation
    with a Kolmogorov-Arnold Network (KAN) layer.
    """
    def __init__(self, in_channels, out_channels):
        super(GKANConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.kan = KANLinear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: Apply KANLinear transformation
        # Replaces the linear transformation (W) in standard GCN.
        x = self.kan(x)

        # Step 4: Propagate messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Normalize node features.
        return norm.view(-1, 1) * x_j

class GKAN(nn.Module):
    """
    Graph Neural Network using KAN layers for feature transformation and classification.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GKAN, self).__init__()
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GKANConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GKANConv(hidden_channels, hidden_channels))

        # Global pooling
        self.pool = global_mean_pool

        # Final classification head (KAN)
        # Replacing MLP with KAN means: hidden -> out (3 classes)
        self.head = KANLinear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            # KAN includes activation (SiLU by default) inside KANLinear,
            # so the output of GKANConv is already non-linear transformed and aggregated.
            # We don't apply extra ReLU here.

        # Global pooling
        # Aggregates node features into a graph-level embedding
        x = self.pool(x, batch)

        # Classification head
        x = self.head(x)

        return x
