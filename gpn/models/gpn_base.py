from typing import Dict, Tuple, List
from gpn.utils.config import ModelConfiguration
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.utils as tu
from torch_geometric.data import Data
from gpn.nn import uce_loss, entropy_reg
from gpn.layers import APPNPPropagation, LinearSequentialLayer
from gpn.utils import apply_mask
from gpn.utils import Prediction, ModelConfiguration
from gpn.layers import Density, Evidence, ConnectedComponents
from .model import Model


class GPN(Model):
    """Graph Posterior Network model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)

        if self.params.num_layers is None:
            num_layers = 0

        else:
            num_layers = self.params.num_layers

        if num_layers > 2:
            self.input_encoder = LinearSequentialLayer(
                self.params.dim_features,
                [self.params.dim_hidden] * (num_layers - 2),
                self.params.dim_hidden,
                batch_norm=self.params.batch_norm,
                dropout_prob=self.params.dropout_prob,
                activation_in_all_layers=True)
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(self.params.dim_features, self.params.dim_hidden),
                nn.ReLU(),
                nn.Dropout(p=self.params.dropout_prob))

        self.latent_encoder = nn.Linear(self.params.dim_hidden, self.params.dim_latent)

        use_batched = True if self.params.use_batched_flow else False 
        self.flow = Density(
            dim_latent=self.params.dim_latent,
            num_mixture_elements=self.params.num_classes,
            radial_layers=self.params.radial_layers,
            maf_layers=self.params.maf_layers,
            gaussian_layers=self.params.gaussian_layers,
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params.alpha_evidence_scale)

        self.propagation = APPNPPropagation(
            K=self.params.K,
            alpha=self.params.alpha_teleport,
            add_self_loops=self.params.add_self_loops,
            cached=False,
            normalization='sym')

        assert self.params.pre_train_mode in ('encoder', 'flow', None)
        assert self.params.likelihood_type in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', None)

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        h = self.input_encoder(data.x)
        z = self.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.get_class_probalities(data)
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if '-plus-classes' in self.params.alpha_evidence_scale:
            further_scale = self.params.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.params.dim_latent,
            further_scale=further_scale).exp()

        alpha_features = 1.0 + beta_ft

        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        soft = alpha / alpha.sum(-1, keepdim=True)
        logits = None
        log_soft = soft.log()

        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha,
            soft=soft,
            log_soft=log_soft,
            hard=hard,

            logits=logits,
            latent=z,
            latent_features=z,

            hidden=h,
            hidden_features=h,

            evidence=beta.sum(-1),
            evidence_ft=beta_ft.sum(-1),
            log_ft_per_class=log_q_ft_per_class,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def get_optimizer(self, lr: float, weight_decay: float) -> Tuple[optim.Adam, optim.Adam]:
        flow_lr = lr if self.params.factor_flow_lr is None else self.params.factor_flow_lr * lr
        flow_weight_decay = weight_decay if self.params.flow_weight_decay is None else self.params.flow_weight_decay

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f'flow.{p[0]}' for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = optim.Adam(flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay)
        model_optimizer = optim.Adam(
            [{'params': flow_param_weights, 'lr': flow_lr, 'weight_decay': flow_weight_decay},
             {'params': params}],
            lr=lr, weight_decay=weight_decay)

        return model_optimizer, flow_optimizer

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

        if self.params.pre_train_mode == 'encoder':
            warmup_optimizer = model_optimizer
        else:
            warmup_optimizer = flow_optimizer

        return warmup_optimizer

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        # similar to warmup
        return self.get_warmup_optimizer(lr, weight_decay)

    def uce_loss(self, prediction: Prediction, data: Data, approximate=True) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_train, y = apply_mask(data, prediction.alpha, split='train')
        reg = self.params.entropy_reg
        return uce_loss(alpha_train, y, reduction='sum'), \
            entropy_reg(alpha_train, reg, approximate=approximate, reduction='sum')

    def loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        uce, reg = self.uce_loss(prediction, data)
        n_train = data.train_mask.sum() if self.params.loss_reduction == 'mean' else 1
        return {'UCE': uce / n_train, 'REG': reg / n_train}

    def warmup_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        if self.params.pre_train_mode == 'encoder':
            return self.CE_loss(prediction, data)

        return self.loss(prediction, data)

    def finetune_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        return self.warmup_loss(prediction, data)

    def likelihood(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_class_probalities(self, data: Data) -> torch.Tensor:
        l_c = torch.zeros(self.params.num_classes, device=data.x.device)
        y_train = data.y[data.train_mask]

        # calculate class_counts L(c)
        for c in range(self.params.num_classes):
            class_count = (y_train == c).int().sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c
