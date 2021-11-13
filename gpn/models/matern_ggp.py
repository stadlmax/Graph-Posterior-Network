from typing import Tuple
import torch
import os
import tensorflow as tf
import networkx as nx
import scipy as sp
import numpy as np
import torch_geometric.utils as tu
from torch_geometric.data import Data
import gpflow

from gpn.utils import ModelConfiguration
from .gpflow_gpp import GPFLOWGGP

from .matern_ggp_utils import GPInducingVariables, GraphMaternKernel, optimize_SVGP

gpflow.config.set_default_float(tf.float64)
gpflow.config.set_default_summary_fmt("notebook")
tf.get_logger().setLevel('ERROR')


class MaternGGP(GPFLOWGGP):
    """model wrapping MaternGGP into our pipeline
    
    code taken from https://github.com/spbu-math-cs/Graph-Gaussian-Processes
    """
    def __init__(self, params: ModelConfiguration):
        super().__init__(params)

        self.nu = 3/2
        self.kappa = 5
        self.sigma_f = 1.0

        self.epochs = 20_000
        self.learning_rate = 0.001
        self.num_eigenpairs = 500

    def _train_model(self, data: Data) -> None:
        num_classes = self.params.num_classes
        num_train = data.train_mask.sum().item()
        dtype = tf.float64

        x_id_all = torch.arange(data.x.size(0)).double().view(-1, 1)
        y_all = data.y.double()

        x_train = x_id_all[data.train_mask].cpu().numpy()
        y_train = y_all[data.train_mask].cpu().numpy()
        x_id_all = x_id_all.cpu().numpy()
        y_all = y_all.cpu().numpy()
        data_train = (x_train, y_train)

        eigen_dir = os.path.join(os.getcwd(), 'saved_experiments', 'uncertainty_experiments')
        eigen_dir = os.path.join(eigen_dir, 'eigenpairs', self.storage_params['dataset'])

        if os.path.exists(eigen_dir):
            eigenvalues = tf.convert_to_tensor(np.load(
                os.path.join(eigen_dir, 'eigenvalues.npy'), allow_pickle=False))
            eigenvectors = tf.convert_to_tensor(
                np.load(os.path.join(eigen_dir, 'eigenvectors.npy'), allow_pickle=False))

        else:
            os.makedirs(eigen_dir)

            G = tu.to_networkx(data, to_undirected=True)
            laplacian = sp.sparse.csr_matrix(nx.laplacian_matrix(G), dtype=np.float64)
            if self.num_eigenpairs >= len(G):
                num_eigenpairs = len(G)
            else:
                num_eigenpairs = self.num_eigenpairs

            eigenvalues, eigenvectors = tf.linalg.eigh(laplacian.toarray())
            eigenvectors, eigenvalues = eigenvectors[:, :num_eigenpairs], eigenvalues[:num_eigenpairs]

            np.save(os.path.join(eigen_dir, 'eigenvalues.npy'), eigenvalues.numpy(), allow_pickle=False)
            np.save(os.path.join(eigen_dir, 'eigenvectors.npy'), eigenvectors.numpy(), allow_pickle=False)

            eigenvalues = tf.convert_to_tensor(eigenvalues, dtype=dtype)
            eigenvectors = tf.convert_to_tensor(eigenvectors, dtype)

        inducing_points = GPInducingVariables(x_train)

        kernel = GraphMaternKernel(
            (eigenvectors, eigenvalues), nu=self.nu, kappa=self.kappa, sigma_f=self.sigma_f,
            vertex_dim=0, point_kernel=None, dtype=dtype)

        model = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=gpflow.likelihoods.MultiClass(num_classes),
            inducing_variable=inducing_points,
            num_latent_gps=num_classes,
            whiten=True,
            q_diag=True,
        )

        adam_opt = tf.optimizers.Adam(self.learning_rate)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=self.learning_rate)

        optimize_SVGP(model, (adam_opt, natgrad_opt), self.epochs, data_train, num_train, True)
        self.model = model

    def _predict(self, data: Data) -> Tuple[np.array, np.array]:
        x_id_all = torch.arange(data.x.size(0)).double().view(-1, 1).cpu().numpy()
        mean, var = self.model.predict_y(x_id_all)
        return mean, var
