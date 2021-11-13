"""
https://github.com/spbu-math-cs/Graph-Gaussian-Processes
"""

import numpy as np
import tensorflow as tf
from tqdm.autonotebook import trange
import gpflow
from gpflow.inducing_variables import InducingVariables
from gpflow.base import TensorLike
from gpflow import covariances as cov
from gpflow.kernels import Kernel


class GraphMaternKernel(gpflow.kernels.Kernel):
    """The Matern kernel on Graph. Kernel is direct product of Matern Kernel on Graph and some kernel on \R^d

    Attributes
    ----------

    eigenpairs : tuple
        Truncated tuple returned by tf.linalg.eigh applied to the Laplacian of the graph.
    nu : float
        Trainable smoothness hyperparameter.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    vertex_dim: int
        dimension of \R^d
    point_kernel: gpflow.kernels.Kernel
        kernel on \R^d
    active_dims: slice or list of indices
        gpflow.kernel.Kernel parameter.
    dtype : tf.dtypes.DType
        type of tensors, tf.float64 by default
        """

    def __init__(self, eigenpairs, nu=3, kappa=4, sigma_f=1,
                 vertex_dim=0, point_kernel=None, active_dims=None, dtype=tf.float64):

        self.eigenvectors, self.eigenvalues = eigenpairs
        #pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        self.num_verticies = tf.cast(tf.shape(self.eigenvectors)[0], dtype=dtype)
        self.vertex_dim = vertex_dim
        if vertex_dim != 0:
            self.point_kernel = point_kernel
        else:
            self.point_kernel = None
        self.dtype = dtype

        self.nu = gpflow.Parameter(
            nu, dtype=self.dtype, transform=gpflow.utilities.positive(), name='nu')
        self.kappa = gpflow.Parameter(
            kappa, dtype=self.dtype, transform=gpflow.utilities.positive(), name='kappa')
        self.sigma_f = gpflow.Parameter(
            sigma_f, dtype=self.dtype, transform=gpflow.utilities.positive(), name='sigma_f')

        super().__init__(active_dims=active_dims)

    def eval_S(self, kappa, sigma_f):
        #pylint: disable=invalid-unary-operand-type
        S = tf.pow(self.eigenvalues + 2 * self.nu / kappa / kappa, -self.nu)
        S = tf.multiply(S, self.num_verticies / tf.reduce_sum(S))
        S = tf.multiply(S, sigma_f)
        return S

    def _eval_K_vertex(self, X_id, X2_id):
        if X2_id is None:
            X2_id = X_id

        S = self.eval_S(self.kappa, self.sigma_f)

        K_vertex = (tf.gather_nd(self.eigenvectors, X_id) * S[None, :]) @ \
            tf.transpose(tf.gather_nd(self.eigenvectors, X2_id))  # shape (n,n)

        return K_vertex

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        #pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        if self.vertex_dim == 0:
            X_id = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1])
            X2_id = tf.reshape(tf.cast(X2[:, 0], dtype=tf.int32), [-1, 1])
            K = self._eval_K_vertex(X_id, X2_id)
        else:
            X_id, X_v = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1]), X[:, 1:]
            X2_id, X2_v = tf.reshape(tf.cast(X2[:, 0], dtype=tf.int32), [-1, 1]), X2[:, 1:]

            K_vertex = self._eval_K_vertex(X_id, X2_id)
            K_point = self.point_kernel.K(X_v, X2_v)

            K = tf.multiply(K_point, K_vertex)

        return K

    def _eval_K_diag_vertex(self, X_id):
        S = self.eval_S(self.kappa, self.sigma_f)

        K_diag_vertex = tf.reduce_sum(tf.transpose((tf.gather_nd(self.eigenvectors, X_id)) * S[None, :]) *
                                      tf.transpose(tf.gather_nd(self.eigenvectors, X_id)), axis=0)

        return K_diag_vertex

    def K_diag(self, X):
        #pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        if self.vertex_dim == 0:
            X_id = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1])
            K_diag = self._eval_K_diag_vertex(X_id)
        else:
            X_id, X_v = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1]), X[:, 1:]
            K_diag_vertex = self._eval_K_diag_vertex(X_id)
            K_diag_point = self.point_kernel.K_diag(X_v)
            K_diag = K_diag_point * K_diag_vertex
        return K_diag

    def sample(self, X):
        K_chol = tf_jitchol(self.K(X), dtype=self.dtype)
        sample = K_chol.dot(np.random.randn(tf.shape(K_chol)[0]))
        return sample


class GPInducingVariables(InducingVariables):
    """
       Graph inducing points.
       The first coordinate is vertex index.
       Other coordinates are matched with points on \R^d.
       Note that vertex indices are not trainable.
    """
    def __init__(self, x):
        self.x_id = x[:, :1]
        if len(x.shape) > 1:
            self.x_v = gpflow.Parameter(x[:, 1:], dtype=gpflow.default_float())

        self.N = self.x_id.shape[0]

    def __len__(self):

        return self.N

    @property
    def GP_IV(self):
        #pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return tf.concat([self.x_id, self.x_v], axis=1)


@cov.Kuu.register(GPInducingVariables, gpflow.kernels.Kernel)
def Kuu_kernel_GPinducingvariables(
        inducing_variable: InducingVariables,
        kernel: Kernel,
        jitter=0.0):
    GP_IV = inducing_variable.GP_IV

    Kuu = kernel.K(GP_IV)
    Kuu += jitter * tf.eye(tf.shape(Kuu)[0], dtype=Kuu.dtype)

    return Kuu


@cov.Kuf.register(GPInducingVariables, gpflow.kernels.Kernel, TensorLike)
def Kuf_kernel_GPinducingvariables(
        inducing_variable: InducingVariables,
        kernel: Kernel,
        X: tf.Tensor):
    GP_IV = inducing_variable.GP_IV

    Kuf = kernel.K(GP_IV, X)

    return Kuf


def tf_jitchol(mat, jitter=0, dtype=tf.float32):
    """Run Cholesky decomposition with an increasing jitter,
    until the jitter becomes too large.
    Arguments
    ---------
    mat : (m, m) tf.Tensor
        Positive-definite matrix
    jitter : float
        Initial jitter
    """
    try:
        chol = tf.linalg.cholesky(mat)
        return chol
    except:
        new_jitter = jitter*10.0 if jitter > 0.0 else 1e-15
        if new_jitter > 1.0:
            raise RuntimeError('Matrix not positive definite even with jitter')

        #pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        new_jitter = tf.cast(new_jitter, dtype=dtype)
        return tf_jitchol(mat + tf.multiply(new_jitter, tf.eye(mat.shape[-1], dtype=dtype)), new_jitter)


def opt_step(opt, loss, variables):
    opt.minimize(loss, var_list=variables)


def optimize_SVGP(model, optimizers, steps, data_train, train_num, q_diag=True):
    if not q_diag:
        gpflow.set_trainable(model.q_mu, False)
        gpflow.set_trainable(model.q_sqrt, False)

    adam_opt, natgrad_opt = optimizers

    variational_params = [(model.q_mu, model.q_sqrt)]

    autotune = tf.data.experimental.AUTOTUNE
    data_minibatch = (
        tf.data.Dataset.from_tensor_slices(data_train)
        .prefetch(autotune)
        .repeat()
        .shuffle(train_num)
        .batch(train_num)
    )
    data_minibatch_it = iter(data_minibatch)
    loss = model.training_loss_closure(data_minibatch_it)
    adam_params = model.trainable_variables
    natgrad_params = variational_params

    adam_opt.minimize(loss, var_list=adam_params)
    if not q_diag:
        natgrad_opt.minimize(loss, var_list=natgrad_params)
    t = trange(steps)
    for step in t:
        opt_step(adam_opt, loss, adam_params)
        if not q_diag:
            opt_step(natgrad_opt, loss, natgrad_params)
        if step % 200 == 0:
            likelihood = model.elbo(data_train)
            t.set_postfix({'ELBO': likelihood.numpy()})
