import torch
from torch_geometric.data import Data
from gpn.utils import Prediction
from .model import Model


class EnergyScoring(Model):
    """Wrapper for existing models estimating uncertainty through energy scoring given a certain temperature"""

    def __init__(self, model: Model, temperature: float = 1.0):
        super().__init__(None)
        self.model = model
        self.temp = temperature

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data) -> Prediction:
        pred = self.model(data)
        energy = -self.temp * torch.logsumexp(pred.logits / self.temp, dim=-1)
        pred.set_values(
            energy=energy,
            sample_confidence_epistemic=-energy
        )

        if pred.logits_features is not None:
            logits_features = pred.logits_features
            energy_features = -self.temp * torch.logsumexp(logits_features / self.temp, dim=-1)
            pred.set_values(
                energy_features=energy_features,
                sample_confidence_features=-energy_features
            )

        return pred

    def load_from_storage(self):
        raise NotImplementedError

    def save_to_storage(self):
        raise NotImplementedError

    def create_storage(self, *args, **kwargs):
        raise NotImplementedError
