from gpn.utils.config import ModelConfiguration
import torch
from torch_geometric.data import Data
from gpn.utils import Prediction
from gpn.layers import CertaintyDiffusion
from .model import Model


class DiffusionRho(Model):
    """simple parameterless diffusion model based on a proxy of training node density in neighborhoods"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.certainty_diffusion = CertaintyDiffusion(
            self.params.num_classes,
            K=10,
            alpha=0.1,
            add_self_loops=True,
            cached=False,
            normalization='sym',
            w=0.9)

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        p_uc, p_u, p_c = self.certainty_diffusion(data)

        alpha = 1.0 + p_uc
        soft = alpha / alpha.sum(-1, keepdim=True)
        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            hard=hard,
            alpha=alpha,
            p_uc=p_uc,
            p_u=p_u,
            p_c=p_c,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=p_uc[torch.arange(hard.size(0)), hard],

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=None,
            sample_confidence_structure=p_u
        )
        # ---------------------------------------------------------------------------------



        return pred

    def expects_training(self) -> bool:
        return False
