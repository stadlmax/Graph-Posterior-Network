from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from gpn.layers import BayesianGCNConv
from gpn.utils import Prediction, apply_mask, ModelConfiguration
from .model import Model


class BayesianGCN(Model):
    """Bayesian GCN model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.samples = params.bayesian_samples
        self.var_eps = 1.0e-8

        self.conv_1 = BayesianGCNConv(
            params.dim_features, params.dim_hidden,
            pi=params.pi, sigma_1=params.sigma_1,
            sigma_2=params.sigma_2)

        self.conv_2 = BayesianGCNConv(
            params.dim_hidden, params.num_classes,
            pi=params.pi, sigma_1=params.sigma_1,
            sigma_2=params.sigma_2)

        activation = []
        activation.append(nn.ReLU())
        self.activation = nn.Sequential(*activation)

    def forward_impl(self, data: Data, sample: bool = False, calculate_log_probs: bool = False) -> torch.Tensor:
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        x = self.conv_1(
            data.x, edge_index,
            sample=sample, calculate_log_probs=calculate_log_probs)

        x = self.activation(x)

        x = self.conv_2(
            x, edge_index,
            sample=sample, calculate_log_probs=calculate_log_probs)

        return torch.softmax(x, dim=-1)

    def forward(self, data: Data, sample: bool = True, calculate_log_probs: bool = False) -> Prediction:
        samples = 1 if not sample else self.samples
        num_nodes = data.x.size(0)

        softs = torch.zeros(num_nodes, samples, self.params.num_classes).to(data.x.device)
        log_priors = torch.zeros(samples).to(data.x.device)
        log_qs = torch.zeros(samples).to(data.x.device)

        for i in range(self.samples):
            softs[:, i, :] = self.forward_impl(
                data, sample=sample, calculate_log_probs=calculate_log_probs)

            if self.training or calculate_log_probs:
                log_priors[i] = self.log_prior()
                log_qs[i] = self.log_q()

        log_prior = log_priors.mean()
        log_q = log_qs.mean()

        soft = softs.mean(1)

        log_soft = torch.log(soft + 1.0e-10)
        max_soft, hard = soft.max(-1)

        # empirical variance p_c^{(i)} of predicted class
        var = torch.var(softs, dim=-2)
        var_predicted = var[torch.arange(hard.size(0)), hard]

        # empirical variance of p^{(i)}, i.e. overall variance
        # var is tr of var_c
        var = var.sum(-1)
        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            var=var,
            var_predicted=var_predicted,
            softs=softs,

            log_prior=log_prior.view(1),
            log_q=log_q.view(1),

            # prediction confidence
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=1.0 / (var_predicted + self.var_eps),
            prediction_confidence_structure=None,

            # sample confidence
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=1.0 / (var + self.var_eps),
            sample_confidence_features=None,
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def log_prior(self) -> torch.Tensor:
        return self.conv_1.log_prior + self.conv_2.log_prior

    def log_q(self) -> torch.Tensor:
        return self.conv_1.log_q + self.conv_2.log_q

    def loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        return {**self.ELBO_loss(prediction, data)}

    def ELBO_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        # NLL from data
        log_softs = torch.log(1.0e-10 + prediction.softs)
        nll = 0
        for s in range(self.samples):
            y_hat = log_softs[:, s, :]
            y_hat, y = apply_mask(data, y_hat, split='train')
            nll += F.nll_loss(y_hat, y)
        nll = nll / self.samples
        # likelihood-term coming from weights
        #loss = prediction.log_q - prediction.log_prior + nll
        return {'log_q': prediction.log_q, 'log_prior': -prediction.log_prior, 'NLL': nll}
