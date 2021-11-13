from typing import Tuple
import torch
import tensorflow as tf
import networkx as nx
import numpy as np
import torch_geometric.utils as tu
from tqdm.autonotebook import trange
from torch_geometric.data import Data
import gpflow
from gpflow.mean_functions import Constant
from scipy.cluster.vq import kmeans2
from sklearn.feature_extraction.text import TfidfTransformer

from gpn.utils import ModelConfiguration
from .gpflow_gpp import GPFLOWGGP
from .ggp_utils import GraphPolynomial, GraphSVGP, NodeInducingPoints, training_step

gpflow.config.set_default_float(tf.float64)
gpflow.config.set_default_summary_fmt("notebook")
tf.get_logger().setLevel('ERROR')


class GGP(GPFLOWGGP):
    """Graph-Gaussian-Process

    code taken from https://github.com/FelixOpolka/GGP-TF2/blob/master/GraphSVGP.py
    """

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.epochs = 2_000

    def _train_model(self, data: Data) -> None:
        # convert adj-matrix
        G = tu.to_networkx(data, to_undirected=True)
        adj_matrix = nx.adjacency_matrix(G).astype(np.float64)
        # convert features
        node_feats = data.x.cpu().numpy().astype(np.float64)
        transformer = TfidfTransformer(smooth_idf=True)
        node_feats = transformer.fit_transform(node_feats).toarray().astype(np.float64)

        # num-classes
        num_classes = self.params.num_classes

        x_id_all = torch.arange(data.x.size(0)).int()
        node_labels = data.y.int().cpu().numpy()

        idx_train = x_id_all[data.train_mask].cpu().numpy()
        idx_train = tf.constant(idx_train)

        # Init kernel
        kernel = GraphPolynomial(adj_matrix, node_feats, idx_train)

        # Init inducing points
        # use as many inducing points as training samples
        inducing_points = kmeans2(node_feats, len(idx_train), minit='points')[0]
        inducing_points = NodeInducingPoints(inducing_points)

        # Init GP model
        mean_function = Constant()
        gprocess = GraphSVGP(
            kernel, gpflow.likelihoods.MultiClass(num_classes),
            inducing_points, mean_function=mean_function,
            num_latent_gps=num_classes, whiten=True, q_diag=False)

        # Init optimizer
        optimizer = tf.optimizers.Adam()

        t = trange(self.epochs)
        for step in t:
            elbo = -training_step(
                idx_train, node_labels[idx_train], optimizer,
                gprocess)

            if step % 200 == 0:
                t.set_postfix({'ELBO': elbo.numpy()})

        self.model = gprocess

    def _predict(self, data: Data) -> Tuple[np.array, np.array]:
        x_id_all = torch.arange(data.x.size(0)).double().view(-1, 1).cpu().numpy()
        x_id_all = tf.constant(x_id_all)
        mean, var = self.model.predict_y(x_id_all)
        return mean, var
