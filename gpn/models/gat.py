import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from gpn.utils import ModelConfiguration
from .model import Model


class GAT(Model):
    """GAT model"""

    def __init__(self, params: ModelConfiguration):

        super().__init__(params)

        self.input = nn.Dropout(p=self.params.dropout_prob)
        self.conv1 = GATConv(
            self.params.dim_features, 
            self.params.dim_hidden, 
            heads=self.params.heads_conv1,
            dropout=self.params.coefficient_dropout_prob, 
            negative_slope=self.params.negative_slope)

        activation = []

        activation.append(nn.ELU())
        activation.append(nn.Dropout(p=self.params.dropout_prob))

        self.activation = nn.Sequential(*activation)

        self.conv2 = GATConv(
            self.params.dim_hidden * self.params.heads_conv1,
            self.params.num_classes,
            heads=self.params.heads_conv2,
            concat=False, 
            dropout=self.params.coefficient_dropout_prob,
            negative_slope=self.params.negative_slope)

    def forward_impl(self, data: Data) -> Tensor:
        x = self.input(data.x)
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        x = self.conv1(data.x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x
