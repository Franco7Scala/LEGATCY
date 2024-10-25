import torch
import torch.nn as nn
from torch_geometric.nn import Linear, to_hetero
from torch_geometric.nn.conv import GATv2Conv


#Questa GAT ha due layer. è pensata solo per la classificazione, perché out_channels = num_classes
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv((-1, -1), out_channels, add_self_loops=False, dropout=dropout)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x.relu()) #self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x.relu()) #self.lin2(x)
        return x
