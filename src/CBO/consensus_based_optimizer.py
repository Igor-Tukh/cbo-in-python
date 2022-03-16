import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.optimizers import Optimizer


class CBO(Optimizer):
    def __init__(self, objective, dt, lmbda, sigma, alpha, initial_particles, anisotropic=True, n_batches=None):
        super(CBO, self).__init__('Consensus Based Optimizer')
        self._objective = objective
        self._dt = dt
        self._lambda = lmbda
        self._sigma = sigma
        # In case if we want to use non-constant alpha values
        self._set_hyper('alpha', alpha)
        self._V = initial_particles
        self._anisotropic = anisotropic
        self._noise = tfd.Normal(loc=0., scale=1.)
        self._n_batches = 1 if n_batches is None else n_batches

    def apply_cooling(self, epoch):
        self._set_hyper('alpha', self._get_hyper('alpha') * 2)
        self._sigma = self._sigma * np.log2(epoch + 1) / np.log2(epoch + 2)

    def minimizer(self):
        return self._compute_consensus_point()

    def particles(self):
        return self._V

    def update_objective(self, objective):
        self._objective = objective

    def _compute_consensus_point(self, batch=None):
        V = self._V if batch is None else tf.gather_nd(self._V, indices=batch)
        values = np.array([self._objective(v) for v in V])
        weights = np.exp(-self._get_hyper('alpha') * (values - values.min())).reshape(-1, 1)
        consensus = tf.reshape(tf.reduce_sum(V * weights, axis=0) / weights.sum(), (1, -1))
        return consensus

    def _step(self):
        if self._n_batches == 1:
            batches = np.arange(self._V.shape[0]).reshape(1, -1)
        else:
            batches = np.array_split(np.random.permutation(self._V.shape[0]), self._n_batches)
        for batch in batches:
            V_alpha = self._compute_consensus_point(batch.reshape(-1, 1))
            noise = self._noise.sample(self._V.shape)
            diff = self._V - V_alpha
            noise_weight = tf.abs(diff) if self._anisotropic else tf.reshape(tf.norm(diff, ord=2, axis=1), (-1, 1))
            self._V -= self._lambda * diff * self._dt
            self._V += self._sigma * noise * noise_weight * (self._dt ** 0.5)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        self._step()
        var.assign(tf.reshape(self._compute_consensus_point(), var.shape))

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        self._step()
        var.assign(tf.reshape(self._compute_consensus_point(), var.shape))

    def get_config(self):
        # TODO(itukh): think about the `objective` serialization
        config = super(CBO, self).get_config()
        config.update({
            'dt': self._dt,
            'lmbda': self._lambda,
            'sigma': self._sigma,
            'alpha': self._serialize_hyperparameter('alpha'),
            'anisotropic': self._anisotropic,
            'initial_particles': self._V,
        })
        return config
