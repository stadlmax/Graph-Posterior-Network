from typing import Dict, Tuple
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from gpn.layers import GaussianTransformation
from gpn.layers import GaussianPropagation
from gpn.utils import Prediction, ModelConfiguration

from .model import Model


class RGCN(Model):
    """Robust Graph Convolutional Network model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        gamma = params.gamma

        self.beta_kl = params.beta_kl
        self.beta_reg = params.beta_reg
        self.var_eps = 1.0e-8

        self.gaussian_1 = GaussianTransformation(
            params.dim_features, params.dim_hidden, params.dropout_prob, activation=True)

        # no dropout, no non-linearity: only linear transformation
        self.gaussian_2 = GaussianTransformation(
            params.dim_hidden, params.num_classes, params.dropout_prob, activation=False)

        # propagation
        self.propagation = GaussianPropagation(gamma=gamma)

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t

        # transform features
        mu_1, var_1 = self.gaussian_1(data.x, data.x)
        # propagate
        mu_1p, var_1p = self.propagation(mu_1, var_1, edge_index)

        # transform intermediate representations
        mu_2, var_2 = self.gaussian_2(mu_1p, var_1p)
        # propagate
        mu_2p, var_2p = self.propagation(mu_2, var_2, edge_index)

        # potentially more than 1 sample
        eps = torch.zeros_like(mu_2p).normal_()
        z = mu_2p + eps * torch.sqrt(var_2p + self.var_eps)

        log_soft = F.log_softmax(z, dim=-1)
        soft = torch.exp(log_soft)
        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            logits=z,

            # rgcn
            mu_1=mu_1,
            mu_1p=mu_1p,
            mu_2=mu_2,
            mu_2p=mu_2p,

            var_1=var_1,
            var_1p=var_1p,
            var_2=var_2,
            var_2p=var_2p,

            # confidence of prediction
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=1.0 / (var_2p[torch.arange(hard.size(0)), hard] + self.var_eps),
            prediction_confidence_structure=None,

            # confidence of sample
            sample_confidence_aleatoric=max_soft,
            # trace of covariance matrix
            sample_confidence_epistemic=1.0 / (var_2p.sum(-1) + self.var_eps),
            # trace of covariance matrix (only from features)
            sample_confidence_features=1.0 / (var_1.sum(-1) + self.var_eps),
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def loss(self, prediction: Prediction, data: Data) -> Dict[str, Tensor]:
        # cross entropy
        ce = self.CE_loss(prediction, data)
        # KL regularization
        kl = self.KL(prediction)
        l2 = self.L2()
        return {**ce, **kl, **l2}

    def KL(self, prediction: Prediction) -> Dict[str, Tensor]:
        # for unit gaussian and diagonal covariance matrix
        # KL = 0.5 * sum_i var_i + mu_i^2 - 1 - ln(var_i)
        mu = prediction.mu_1
        var = prediction.var_1
        kl = 0.5 * (var + mu ** 2 - 1 - (var + 1.0e-8).log()).mean(dim=-1).sum()
        return {'KL': self.beta_kl * kl}

    def L2(self) -> Dict[str, Tensor]:
        norm2 = torch.norm(self.gaussian_1.mu.linear.linear.weight, 2).pow(2) + \
            torch.norm(self.gaussian_1.var.linear.linear.weight, 2).pow(2)

        return {'L2': self.beta_reg * norm2}

    def get_optimizer(self, lr: float, weight_decay: float) -> Tuple[optim.Adam, optim.Adam]:
        optimizer = optim.Adam(
            [{'params': self.gaussian_1.parameters(), 'weight_decay': 0.0},
             {'params': self.gaussian_2.parameters(), 'weight_decay': 0.0}],
            lr=lr)

        warmup_optimizer = None

        return optimizer, warmup_optimizer
