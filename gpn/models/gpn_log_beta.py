
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.utils as tu
from gpn.nn import uce_loss, entropy_reg
from gpn.layers import APPNPPropagation
from gpn.utils import apply_mask
from gpn.utils import Prediction
from gpn.layers import Density, Evidence, ConnectedComponents
from .model import Model
from .gpn_base import GPN

class GPN_LOG_BETA(GPN):
    """Graph Posterior Network model (used in ablation studies: diffusing log-beta instead of beta)"""

    def forward(self, data):
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
            further_scale=further_scale)

        alpha = 1.0 + self.propagation(beta_ft, edge_index).exp()

        beta_ft = beta_ft.exp()
        alpha_features = 1.0 + beta_ft

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

            evidence=alpha.sum(-1),
            evidence_ft=alpha_features.sum(-1),
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
