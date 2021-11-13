import torch
import torch.nn as nn
from torch_geometric.data import Data
from gpn.utils import Prediction
from .model import Model


class DropoutEnsemble(Model):
    """DropoutEnsemble


    Wrapper-class for a DropoutEnsemble created from one model where 
    during inference the dropout ensemble is created from multiple forward passes
    """

    def __init__(self, model: Model, num_samples: int = 10):
        super().__init__(None)
        self.model = model
        self.num_samples = num_samples
        self.y_hat = [None] * num_samples
        self.var_eps = 1.0e-8

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        was_training = self.model.training
        self._set_dropout_train(self.num_samples > 1)

        softs = torch.stack([self.model(data).soft for s in range(self.num_samples)], dim=1)

        if not was_training:
            self._set_dropout_train(False)

        soft = softs.mean(1)
        max_soft, hard = soft.max(-1)

        # empirical variance p_c^{(i)} of predicted class
        var = torch.var(softs, dim=-2)
        var_predicted = var[torch.arange(hard.size(0)), hard]

        # empirical variance of p^{(i)}, i.e. overall variance
        # var is tr of var_c
        var = var.sum(-1)

        # ---------------------------------------------------------------------------------
        # prediction and intermediary scores
        pred = Prediction(
            soft=soft,
            hard=hard,
            var=var,
            var_predicted=var_predicted,
            softs=softs,

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

    def _set_dropout_train(self, do_train: bool) -> False:
        # Ensures that all dropout nodes drop neurons
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                # Always set to evaluation
                module.train(do_train)

    def load_from_storage(self):
        raise NotImplementedError

    def save_to_storage(self):
        raise NotImplementedError

    def create_storage(self, *args, **kwargs):
        raise NotImplementedError
