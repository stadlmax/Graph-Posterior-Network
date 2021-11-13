import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from gpn.utils import ModelConfiguration
from gpn.layers import GCNConv
from .model import Model


class GCN(Model):
    """GCN model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)

        self.conv1 = GCNConv(
            self.params.dim_features,
            self.params.dim_hidden,
            cached=False,
            add_self_loops=True,
            normalization='sym')

        activation = []

        activation.append(nn.ReLU())
        activation.append(nn.Dropout(p=self.params.dropout_prob))

        self.activation = nn.Sequential(*activation)

        self.conv2 = GCNConv(
            self.params.dim_hidden,
            self.params.num_classes,
            cached=False,
            add_self_loops=True,
            normalization='sym')

    def forward_impl(self, data: Data) -> Tensor:
        if data.edge_index is not None:
            edge_index = data.edge_index
            if self.params.dropout_prob_adj > 0:
                edge_index, _ = dropout_adj(edge_index, p=self.params.dropout_prob_adj, 
                                            force_undirected=False, training=self.training)
        else:
            edge_index = data.adj_t

        x = self.conv1(data.x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x
