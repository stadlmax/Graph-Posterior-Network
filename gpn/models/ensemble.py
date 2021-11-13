from typing import List
import torch
import torch.nn as nn
from torch_geometric.data import Data
from gpn.utils import Prediction, ModelConfiguration
from .model import Model
from .utils import create_model


class Ensemble(Model):
    """Ensemble

    Wrapper-Class for an ensemble which initializes the samle model N-times by
    aggregating predictions and allowing for parallel training
    """

    def __init__(self, params: ModelConfiguration, models: List[Model]):
        super().__init__(params)

        self.models = nn.ModuleList(models)
        self.num_models = len(models) if models is not None else -1
        self.var_eps = 1.0e-8

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        softs = torch.stack([m(data).soft for m in self.models], dim=1)

        soft = softs.mean(1)
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

    def save_to_storage(self) -> None:
        raise NotImplementedError

    def load_from_storage(self) -> None:
        if self.storage is None:
            raise RuntimeError('Error on loading model, storage does not exist!')

        path = self.storage.retrieve_model_dir_path(self.params.model_name, self.storage_params)
        models = []

        # ensemble assumes that random initializations 1 - 10 are in path
        min_init_no = self.params.ensemble_min_init_no
        max_init_no = self.params.ensemble_max_init_no

        for i in range(min_init_no, max_init_no):
            model = create_model(self.params)
            model_file_path = self.storage.build_model_file_path(path, init_no=i)
            model.load_from_file(model_file_path)
            models.append(model)

        self.models = nn.ModuleList(models)
