from typing import Dict
import copy
from gpn.experiments.dataset import ExperimentDataset
import torch
import torch.distributions as D
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from sacred import Experiment
import gpn.nn as unn
from gpn.layers import GCNConv
from gpn.utils import Prediction, apply_mask
from gpn.utils import RunConfiguration, ModelConfiguration, DataConfiguration
from gpn.utils import TrainingConfiguration
from gpn.nn import loss_reduce
from .model import Model
from .gdk import GDK
from .gcn import GCN


class SGCN(Model):
    """SGCN or GDKE-GCN model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.alpha_prior = None
        self.y_teacher = None

        self.conv1 = GCNConv(
            self.params.dim_features,
            self.params.dim_hidden,
            cached=False,
            add_self_loops=True,
            normalization='sym')

        activation = []

        activation.append(nn.ReLU())
        activation.append(nn.Dropout(p=self.params.dropout_prob))

        self.activation = nn.Sequential(*activation)

        self.conv2 = GCNConv(
            self.params.dim_hidden,
            self.params.num_classes,
            cached=False,
            add_self_loops=True,
            normalization='sym')

        self.evidence_activation = torch.exp
        self.epoch = None

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        if self.training or (not self.params.use_bayesian_dropout):
            x = self.conv1(data.x, edge_index)
            x = self.activation(x)
            x = self.conv2(x, edge_index)
            evidence = self.evidence_activation(x)

        else:
            self_training = self.training
            self.train()
            samples = [None] * self.params.num_samples_dropout

            for i in range(self.params.num_samples_dropout):
                x = self.conv1(data.x, edge_index)
                x = self.activation(x)
                x = self.conv2(x, edge_index)
                samples[i] = x

            log_evidence = torch.stack(samples, dim=1)

            if self.params.sample_method == 'log_evidence':
                log_evidence = log_evidence.mean(dim=1)
                evidence = self.evidence_activation(log_evidence)

            elif self.params.sample_method == 'alpha':
                evidence = self.evidence_activation(log_evidence)
                evidence = evidence.mean(dim=1)

            else:
                raise AssertionError

            if self_training:
                self.train()
            else:
                self.eval()

        alpha = 1.0 + evidence
        soft = alpha / alpha.sum(-1, keepdim=True)
        max_soft, hard = soft.max(-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            hard=hard,
            alpha=alpha,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=None,
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def loss(self, prediction: Prediction, data: Data) -> Dict[str, Tensor]:

        if self.params.loss_reduction in ('sum', None):
            n_nodes = 1.0
            frac_train = 1.0

        else:
            n_nodes = data.y.size(0)
            frac_train = data.train_mask.float().mean()

        alpha = prediction.alpha
        #n_nodes = data.y.size(0)
        #n_train = data.train_mask.sum()
        # bayesian risk of sum of squares
        alpha_train, y = apply_mask(data, alpha, split='train')
        bay_risk = unn.bayesian_risk_sosq(alpha_train, y, reduction='sum')
        losses = {'BR': bay_risk * 1.0 / (n_nodes * frac_train)}

        # KL divergence w.r.t. alpha-prior from Gaussian Dirichlet Kernel
        if self.params.use_kernel:
            dirichlet = D.Dirichlet(alpha)
            alpha_prior = self.alpha_prior.to(alpha.device)
            dirichlet_prior = D.Dirichlet(alpha_prior)
            KL_prior = D.kl.kl_divergence(dirichlet, dirichlet_prior)
            KL_prior = loss_reduce(KL_prior, reduction='sum')
            losses['KL_prior'] = self.params.lambda_1 * KL_prior / n_nodes

        # KL divergence for teacher training
        if self.params.teacher_training:
            assert self.y_teacher is not None

            # currently only works for full-batch training
            # i.e. epochs == iterations
            if self.training:
                if self.epoch is None:
                    self.epoch = 0
                else:
                    self.epoch += 1

            y_teacher = self.y_teacher.to(prediction.soft.device)
            lambda_2 = min(1.0, self.epoch * 1.0 / 200)
            categorical_pred = D.Categorical(prediction.soft)
            categorical_teacher = D.Categorical(y_teacher)
            KL_teacher = D.kl.kl_divergence(categorical_pred, categorical_teacher)
            KL_teacher = loss_reduce(KL_teacher, reduction='sum')
            losses['KL_teacher'] = lambda_2 * KL_teacher / n_nodes

        return losses

    def create_storage(self, run_cfg: RunConfiguration, data_cfg: DataConfiguration,
                       model_cfg: ModelConfiguration, train_cfg: TrainingConfiguration,
                       ex: Experiment = None):
        # create storage for model itself
        super().create_storage(run_cfg, data_cfg, model_cfg, train_cfg, ex=ex)

        if run_cfg.job == 'train':
            # create kernel and load alpha-prior
            gdk_config = ModelConfiguration(
                model_name='GDK',
                num_classes=model_cfg.num_classes,
                dim_features=model_cfg.dim_features,
                seed=model_cfg.seed,
                init_no=1 # GDK only with init_no = 1
            )
            kernel = GDK(gdk_config)

            if data_cfg.ood_flag and data_cfg.ood_type == 'leave_out_classes_evasion':
                dataset = ExperimentDataset(data_cfg)
                prediction = kernel(dataset.train_dataset[0])
                alpha_prior = prediction.alpha
                self.alpha_prior = alpha_prior
            else:
                kernel.create_storage(run_cfg, data_cfg, gdk_config, TrainingConfiguration())
                kernel.load_from_storage()
                self.alpha_prior = kernel.cached_alpha

            if model_cfg.teacher_training:
                # create teacher and load y-teacher
                # only works for current vanilla-gcn
                # with default parameters
                gcn_config = ModelConfiguration(
                    model_name='GCN',
                    dim_hidden=64,
                    dropout_prob=0.8,
                    dropout_prob_adj=0.0,
                    dim_features=self.params.dim_features,
                    num_classes=self.params.num_classes,
                    seed=model_cfg.seed,
                    init_no=model_cfg.init_no
                )
                gcn_train_config = copy.deepcopy(train_cfg)
                gcn_train_config.set_values(
                    lr=0.01,
                    weight_decay=0.0001,
                    epochs=100_000,
                    stopping_patience=50
                )

                teacher = GCN(gcn_config)
                teacher.create_storage(run_cfg, data_cfg, gcn_config, gcn_train_config)
                teacher.load_from_storage()

                if data_cfg.ood_flag and data_cfg.ood_type == 'leave_out_classes_evasion':
                    dataset = ExperimentDataset(data_cfg)
                    with torch.no_grad():
                        y_teacher = teacher(dataset.train_dataset[0]).soft
                    self.y_teacher = y_teacher

                else:
                    self.y_teacher = teacher.cached_y
