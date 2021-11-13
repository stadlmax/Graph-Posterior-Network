
from typing import Dict, Tuple
from sacred import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.utils as tu
from torch_geometric.data import Data
from gpn.nn import uce_loss, entropy_reg
from gpn.layers import APPNPPropagation
from gpn.utils import RunConfiguration, DataConfiguration
from gpn.utils import ModelConfiguration, TrainingConfiguration
from gpn.utils import apply_mask
from gpn.utils import Prediction
from gpn.layers import Density, Evidence, ConnectedComponents
from .model import Model
from .gpn_base import GPN
from .gpn_postnet_diff import GPN_MLP


class PostNet(Model):
    """PosteriorNetwork model (used in ablation studies, i.e. only PostNet on feature-level, no propagation)"""

    def __init__(self, params):
        super().__init__(None)
        self.gpn_mlp = GPN_MLP(params)

    def forward(self, data: Data):
        h = self.gpn_mlp.input_encoder(data.x)
        z = self.gpn_mlp.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.gpn_mlp.get_class_probalities(data)
        log_q_ft_per_class = self.gpn_mlp.flow(z) + p_c.view(1, -1).log()

        beta_ft = self.gpn_mlp.evidence(
            log_q_ft_per_class, dim=self.gpn_mlp.params.dim_latent,
            further_scale=self.gpn_mlp.params.num_classes).exp()

        # alpha is uniform prior together with evidence votes
        alpha_features = 1.0 + beta_ft

        soft = alpha_features / alpha_features.sum(-1, keepdim=True)
        logits = None
        log_soft = soft.log()

        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha_features,
            soft=soft,
            log_soft=log_soft,
            hard=hard,

            logits=logits,
            latent=z,
            latent_features=z,

            hidden=h,
            hidden_features=h,

            evidence=beta_ft.sum(-1),
            evidence_ft=beta_ft.sum(-1),
            log_ft_per_class=log_q_ft_per_class,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha_features[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha_features.sum(-1),
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def create_storage(self, run_cfg: RunConfiguration, data_cfg: DataConfiguration,
                       model_cfg: ModelConfiguration, train_cfg: TrainingConfiguration,
                       ex: Experiment = None):
        # create storage for model itself
        postnet_model_cfg = model_cfg.clone()
        postnet_model_cfg.set_values(
            model_name='GPN_MLP'
        )
        self.gpn_mlp.create_storage(run_cfg, data_cfg, postnet_model_cfg, train_cfg, ex=ex)

    def load_from_storage(self):
        self.gpn_mlp.load_from_storage()

    def expects_training(self) -> bool:
        return False

    def loss(self, prediction, data):
        raise NotImplementedError

    def is_finetuning(self) -> bool:
        return False

    def is_warming_up(self) -> bool:
        return False

    def save_to_file(self, model_path: str) -> None:
        raise NotImplementedError

    def load_from_file(self, model_path: str) -> None:
        raise NotImplementedError

    def save_to_storage(self) -> None:
        raise NotImplementedError

    def get_optimizer(self, lr: float, weight_decay: float) -> Tuple[optim.Adam, optim.Adam]:
        raise NotImplementedError

    def warmup_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def finetune_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        raise NotImplementedError

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        raise NotImplementedError
