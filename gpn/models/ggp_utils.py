

from typing import Tuple

import gpflow
import tensorflow as tf
import numpy as np
from gpflow import Parameter
from gpflow.inducing_variables.inducing_variables import InducingPointsBase
from gpflow import covariances as cov
from gpflow.models.svgp import SVGP


def sparse_mat_to_sparse_tensor(sparse_mat):
    """
    Converts a scipy csr_matrix to a tensorflow SparseTensor.
    """
    coo = sparse_mat.tocoo()
    indices = np.stack([coo.row, coo.col], axis=-1)
    tensor = tf.sparse.SparseTensor(indices, sparse_mat.data, sparse_mat.shape)
    return tensor


def get_submatrix(adj_matrix, node_idcs):
    """
    Returns indices of nodes that are neighbors of any of the nodes in
    node_idcs.
    """
    adj_matrix[np.diag_indices(adj_matrix.shape[0])] = 1.0
    sub_mat = adj_matrix[node_idcs, :].tocoo()
    rel_node_idcs = np.unique(sub_mat.col)
    return rel_node_idcs



class GraphSVGP(SVGP):
    """GraphSVGP
    
    code taken from https://github.com/FelixOpolka/GGP-TF2/blob/master/GraphSVGP.py
    """

    def log_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Overrides the default log_likelihood implementation of SVGP to employ
        more efficient computation that is possible because we operate on a
        graph (rather than the Euclidean domain). Otherwise the SVGP produces
        OOM errors.
        """
        kl = self.prior_kl()
        f_mean, f_var = self._predict_f_graph(data[0])
        var_exp = self.likelihood.variational_expectations(f_mean, f_var,
                                                           data[1])
        if self.num_data is not None:
            #pylint: disable=no-member
            scale = self.num_data / tf.shape(self.X)[0]

        else:
            scale = tf.cast(1.0, kl.dtype)

        likelihood = tf.reduce_sum(var_exp) * scale - kl
        return likelihood

    def _predict_f_graph(self, X):
        kernel = self.kernel
        f = self.q_mu
        Z = self.inducing_variable.Z
        num_data = Z.shape[0]  # M
        num_func = f.shape[1]  # K
        Kmn = kernel.Kzx(Z, X)
        Kmm = kernel.Kzz(Z) + tf.eye(num_data, dtype=gpflow.default_float()) * gpflow.default_jitter()
        Lm = tf.linalg.cholesky(Kmm)

        # Compute projection matrix A
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)

        # Compute the covariance due to the conditioning
        f_var = kernel.K_diag_tr() - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
        f_var = tf.tile(tf.expand_dims(f_var, 0), shape)  # Shape [K, N, N] or [K, N]

        # Another backsubstitution in the unwhitened case
        if not self.whiten:
            A = tf.linalg.triangular_solve(tf.transpose(Lm), A, lower=False)

        # Construct the conditional mean
        f_mean = tf.matmul(A, f, transpose_a=True)
        if self.q_sqrt is not None:
            if self.q_sqrt.shape.ndims == 2:
                LTA = A * tf.expand_dims(tf.transpose(self.q_sqrt), 2)  # Shape [K, M, N]
            elif self.q_sqrt.shape.ndims == 3:
                L = tf.linalg.band_part(self.q_sqrt, -1, 0)    # Shape [K, M, M]
                A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
                LTA = tf.matmul(L, A_tiled, transpose_a=True)   # Shape [K, M, N]
            else:
                raise ValueError(f"Bad dimension for q_sqrt: {self.q_sqrt.shape.ndims}")
            f_var = f_var + tf.reduce_sum(tf.square(LTA), 1)    # Shape [K, N]
        f_var = tf.transpose(f_var)     # Shape [N, K] or [N, N, K]
        return f_mean + self.mean_function(X), f_var


def training_step(X_train, y_train, optimizer, gprocess):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(gprocess.trainable_variables)
        objective = -gprocess.elbo((X_train, y_train))
        gradients = tape.gradient(objective, gprocess.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gprocess.trainable_variables))
    return objective


class GraphPolynomial(gpflow.kernels.base.Kernel):
    """
    GraphPolynomial kernel for node classification as introduced in
    Yin Chen Ng, Nicolo Colombo, Ricardo Silva: "Bayesian Semi-supervised
    Learning with Graph Gaussian Processes".
    """

    def __init__(self, sparse_adj_mat, feature_mat, idx_train, degree=3.0,
                 variance=1.0, offset=1.0):
        super().__init__(None)
        self.degree = degree
        self.offset = Parameter(offset, transform=gpflow.utilities.positive())
        self.variance = Parameter(variance, transform=gpflow.utilities.positive())
        # Pre-compute the P-matrix for transforming the base covariance matrix
        # (c.f. paper for details).
        sparse_adj_mat[np.diag_indices(sparse_adj_mat.shape[0])] = 1.0
        self.sparse_P = sparse_mat_to_sparse_tensor(sparse_adj_mat)
        self.sparse_P = self.sparse_P / sparse_adj_mat.sum(axis=1)
        self.feature_mat = feature_mat
        # Compute data required for efficient computation of training
        # covariance matrix.
        (self.tr_feature_mat, self.tr_sparse_P,
         self.idx_train_relative) = self._compute_train_data(
             sparse_adj_mat, idx_train, feature_mat,
             tf.sparse.to_dense(self.sparse_P).numpy())

    def _compute_train_data(self, adj_matrix, train_idcs, feature_mat,
                            conv_mat):
        """
        Computes all the variables required for computing the covariance matrix
        for training in a computationally efficient way. The idea is to cut out
        those features from the original feature matrix that are required for
        predicting the training labels, which are the training nodes' features
        and their neihbors' features.
        :param adj_matrix: Original dense adjacency matrix of the graph.
        :param train_idcs: Indices of the training nodes.
        :param feature_mat: Original dense feature matrix.
        :param conv_mat: Original matrix used for computing the graph
        convolutions.
        :return: Cut outs of only the relevant nodes.
            - Feature matrix containing features of only the "relevant" nodes,
            i.e. the training nodes and their neighbors. Shape [num_rel,
            num_feats].
            - Convolutional matrix for only the relevant nodes. Shape [num_rel,
            num_rel].
            - Indices of the training nodes within the relevant nodes. Shape
            [num_rel].
        """
        sub_node_idcs = get_submatrix(adj_matrix, train_idcs)
        # Compute indices of actual train nodes (excluding their neighbours)
        # within the sub node indices
        relative_train_idcs = np.isin(sub_node_idcs, train_idcs)
        relative_train_idcs = np.where(relative_train_idcs == True)[0]
        return (feature_mat[sub_node_idcs],
                conv_mat[sub_node_idcs, :][:, sub_node_idcs],
                relative_train_idcs)

    def K(self, X, Y=None, presliced=False):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X

        base_cov = (self.variance * tf.matmul(
            self.feature_mat, self.feature_mat, transpose_b=True) + self.offset) ** self.degree
        covar = tf.sparse.sparse_dense_matmul(self.sparse_P, base_cov)
        covar = tf.sparse.sparse_dense_matmul(self.sparse_P, covar, adjoint_b=True)
        covar = tf.gather(tf.gather(covar, X, axis=0), X2, axis=1)
        return covar

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X))

    def K_diag_tr(self):
        base_cov = (self.variance * tf.matmul(
            self.tr_feature_mat, self.tr_feature_mat, transpose_b=True) + self.offset) ** self.degree

        #pylint: disable=no-member
        if self.sparse:
            covar = tf.sparse.sparse_dense_matmul(self.tr_sparse_P, base_cov)
            covar = tf.sparse.sparse_dense_matmul(self.tr_sparse_P, covar, adjoint_b=True)

        else:
            covar = tf.matmul(self.tr_sparse_P, base_cov)
            covar = tf.matmul(self.tr_sparse_P, covar, adjoint_b=True)
        covar = tf.gather(tf.gather(covar, self.idx_train_relative, axis=0), self.idx_train_relative, axis=1)
        return tf.linalg.diag_part(covar)


class NodeInducingPoints(InducingPointsBase):
    """
    Set of real-valued inducing points. See parent-class for details.
    """
    pass


@cov.Kuu.register(NodeInducingPoints, GraphPolynomial)
def Kuu_graph_polynomial(inducing_variable, kernel, jitter=None):
    """
    Computes the covariance matrix between the inducing points (which are not
    associated with any node).
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :return: Covariance matrix between the inducing variables.
    """
    Z = inducing_variable.Z
    covar = (kernel.variance * (tf.matmul(Z, Z, transpose_b=True)) + kernel.offset) ** kernel.degree
    return covar


@cov.Kuf.register(NodeInducingPoints, GraphPolynomial, tf.Tensor)
def Kuf_graph_polynomial(inducing_variable, kernel, X):
    """
    Computes the covariance matrix between inducing points (which are not
    associated with any node) and normal inputs.
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :param X: Normal inputs. Note, however, that to simplify the
    implementation, we pass in the indices of the nodes rather than their
    features directly.
    :return: Covariance matrix between inducing variables and inputs.
    """
    X = tf.reshape(tf.cast(X, tf.int32), [-1])
    Z = inducing_variable.Z
    base_cov = (kernel.variance * tf.matmul(kernel.feature_mat, Z, adjoint_b=True) + kernel.offset)**kernel.degree
    covar = tf.sparse.sparse_dense_matmul(kernel.sparse_P, base_cov)
    covar = tf.gather(tf.transpose(covar), X, axis=1)
    return covar
