import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Linear, Parameter
import numpy as np
from torch_geometric.utils import add_self_loops, degree

class GraphSpatialAttentionNetwork(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphSpatialAttentionNetwork, self).__init__(aggr='add')  # 使用加法聚合。
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.att = Parameter(torch.Tensor(1, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, pos, edge_index):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply linear transformation
        x = self.lin(x)

        # Start propagating messages.
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j, index, ptr, size_i):
        # Compute the physical distance between nodes based on their positions (latitude and longitude).
        dist = torch.norm(pos_i - pos_j, p=2, dim=-1).view(-1, 1)

        # Use the distance to modulate the attention mechanism.
        x_j = x_j * self.att
        attention_coefficients = F.leaky_relu(x_j, negative_slope=0.2)
        attention_weights = F.softmax(-dist, dim=0)  # Inverse distance for attention

        return attention_weights * attention_coefficients * x_j


# Example usage
if __name__ == "__main__":
    num_nodes = 10
    num_features = 5
    num_classes = 2

    # Random graph data
    x = torch.randn(num_nodes, num_features)  # Node features
    pos = torch.randn(num_nodes, 2)  # Node positions (latitude and longitude)
    edge_index = torch.randint(0, num_nodes, (2, 20))  # 20 edges
    print(edge_index.shape)

    # model = GraphSpatialAttentionNetwork(num_features, num_classes)
    # out = model(x, pos, edge_index)
    # print(out.shape)  # Should be [num_nodes, num_classes]

    # visualize(model, x, pos, edge_index)
