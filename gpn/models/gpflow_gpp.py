from typing import Dict, Tuple
import torch
import numpy as np
import torch.optim as optim
import gpflow
import tensorflow as tf
from torch_geometric.data import Data
from gpn.utils import ModelNotFoundError, Prediction, ModelConfiguration
from .model import Model


class GPFLOWGGP(Model):
    """model wrapping a Graph Gaussian Process into our pipeline"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.model = None

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        try:
            self.load_from_storage()
        except ModelNotFoundError:
            self._train_model(data)

        mean, var = self._predict(data)
        soft = torch.from_numpy(mean.numpy()).to(data.x.device)
        var_per_class = torch.from_numpy(var.numpy()).to(data.x.device)

        max_soft, hard = soft.max(dim=-1)
        var_eps = 1.0e-8
        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            hard=hard,
            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=1.0 / (var_per_class[torch.arange(hard.size(0)), hard] + var_eps),
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=1.0 / (var_per_class.sum(-1) + var_eps),
            sample_confidence_features=None,
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def _predict(self, data: Data) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    def _train_model(self, data: Data) -> None:
        raise NotImplementedError

    # -----------------------------------------------------------
    # training is wrapped into "forward-call"
    # i.e. directly for evaluation
    def expects_training(self) -> bool:
        return False

    # no parameters pro-forma
    def get_num_params(self) -> int:
        return 0

    # for compatibility
    def set_expects_training(self, flag: bool) -> None:
        self._expects_training = False

    # -----------------------------------------------------------
    # storing / loading model
    # (e.g. for evasion experiments)
    def save_to_file(self, model_path: str) -> None:
        frozen_model = gpflow.utilities.freeze(self.model)
        module_to_save = tf.Module()
        predict_fn = tf.function(
            frozen_model.predict_y, input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
        )
        module_to_save.predict_y = predict_fn
        tf.saved_model.save(module_to_save, model_path)

    def load_from_file(self, model_path: str) -> None:
        self.model = tf.saved_model.load(model_path)

    # -----------------------------------------------------------
    # as not really run in pipeline
    # those methods are not implemented
    def loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def warmup_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def fintetune_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def CE_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        raise NotImplementedError

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        raise NotImplementedError

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        raise NotImplementedError

    def is_warming_up(self) -> bool:
        raise NotImplementedError

    def is_finetuning(self) -> bool:
        raise NotImplementedError

    def set_warming_up(self, flag: bool) -> None:
        raise NotImplementedError

    def set_finetuning(self, flag: bool) -> None:
        raise NotImplementedError
    # -----------------------------------------------------------
