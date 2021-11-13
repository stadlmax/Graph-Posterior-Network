import torch
from torch_geometric.data import Data
from gpn.layers import LinearSequentialLayer
from gpn.utils import ModelConfiguration
from .model import Model


class MLP(Model):
    """simple node-level MLP model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.linear = LinearSequentialLayer(
            self.params.dim_features,
            self.params.dim_hidden,
            self.params.num_classes,
            dropout_prob=self.params.dropout_prob,
            k_lipschitz=self.params.k_lipschitz,
            num_layers=self.params.num_layers,
            batch_norm=self.params.batch_norm)

    def forward_impl(self, data: Data) -> torch.Tensor:
        return self.linear(data.x)
