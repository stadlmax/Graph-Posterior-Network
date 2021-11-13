from typing import Tuple
import torch
import torch.nn.functional as F
import torch_geometric.nn as tnn
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from gpn.layers import LinearSequentialLayer
from gpn.utils import Prediction, ModelConfiguration
from .model import Model


class APPNP(Model):
    """APPNP model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)

        if params.num_layers is None:
            num_layers = 0

        else:
            num_layers = params.num_layers

        if num_layers > 1:
            dim_hidden = [params.dim_hidden] * (num_layers - 1)
        else:
            dim_hidden = params.dim_hidden

        self.linear = LinearSequentialLayer(
            params.dim_features,
            dim_hidden,
            params.num_classes,
            batch_norm=params.batch_norm,
            dropout_prob=params.dropout_prob)

        self.propagation = tnn.APPNP(
            K=params.K,
            alpha=params.alpha_teleport,
            add_self_loops=params.add_self_loops,
        )

    def forward(self, data: Data) -> Prediction:
        x, h = self.forward_impl(data)

        log_soft = F.log_softmax(x, dim=-1)
        soft = torch.exp(log_soft)
        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            logits=x,
            logits_features=h,
            # confidence of prediction
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=None,
            prediction_confidence_structure=None,

            # confidence of sample
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=None,
            sample_confidence_features=None,
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def forward_impl(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        if data.edge_index is not None:
            edge_index = data.edge_index
            if self.params.dropout_prob_adj > 0:
                edge_index, _ = dropout_adj(edge_index, p=self.params.dropout_prob_adj,
                                            force_undirected=False, training=self.training)
        else:
            edge_index = data.adj_t
        h = self.linear(data.x)
        x = self.propagation(h, edge_index)

        return x, h
