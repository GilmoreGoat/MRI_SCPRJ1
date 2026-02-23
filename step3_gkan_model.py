import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from step2_kan_layer import KANLinear

class GKANConv(MessagePassing):
    """
    Custom Graph Convolution Layer that replaces the standard linear transformation
    with a Kolmogorov-Arnold Network (KAN) layer.
    Updated to "Edge-GKAN" where KAN is applied on edges, incorporating edge attributes.
    """
    def __init__(self, in_channels, out_channels, edge_dim=0):
        super(GKANConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.edge_dim = edge_dim
        # The KAN layer now takes input features + edge features
        self.kan = KANLinear(in_channels + edge_dim, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_dim] or [E]

        # Step 1: Add self-loops to the adjacency matrix.
        # We need to add self-loops to edge_attr as well (with value 1.0 or appropriate)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr,
                                               fill_value=1.0, num_nodes=x.size(0))

        # Step 2: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: Propagate messages.
        # We pass edge_attr to propagate so it reaches message()
        return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)

    def message(self, x_j, norm, edge_attr):
        # x_j has shape [E, in_channels]

        # Prepare input for KAN
        if self.edge_dim > 0:
            if edge_attr is None:
                # If edge_dim is expected but no attribute provided, use default (e.g. 1.0)
                edge_attr = torch.ones(x_j.size(0), self.edge_dim, device=x_j.device, dtype=x_j.dtype)

            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

            # Concatenate node features with edge features
            # This weaves the B-splines into the edge processing
            x_input = torch.cat([x_j, edge_attr], dim=1)
        else:
            x_input = x_j

        # Apply KANLinear transformation on the edges
        out = self.kan(x_input)

        # Normalize node features.
        return norm.view(-1, 1) * out

class EdgeGKAN(nn.Module):
    """
    Graph Neural Network using KAN layers for feature transformation and classification.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, edge_dim=0):
        super(EdgeGKAN, self).__init__()
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GKANConv(in_channels, hidden_channels, edge_dim=edge_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GKANConv(hidden_channels, hidden_channels, edge_dim=edge_dim))

        # Global pooling
        self.pool = global_mean_pool

        # Final classification head (KAN)
        # Replacing MLP with KAN means: hidden -> out (3 classes)
        self.head = KANLinear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            # KAN includes activation (SiLU by default) inside KANLinear.

        # Global pooling
        # Aggregates node features into a graph-level embedding
        x = self.pool(x, batch)

        # Classification head
        x = self.head(x)

        return x
