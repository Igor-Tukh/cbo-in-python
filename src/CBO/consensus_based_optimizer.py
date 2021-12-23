import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.optimizers import Optimizer


class CBO(Optimizer):
    def __init__(self, objective, dt, lmbda, sigma, alpha, initial_particles, anisotropic=True):
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

    def minimizer(self):
        return self._compute_consensus_point()

    def particles(self):
        return self._V

    def _compute_consensus_point(self):
        values = np.array([self._objective(v) for v in self._V])
        weights = np.exp(-self._get_hyper('alpha') * (values - values.min())).reshape(-1, 1)
        return tf.reshape(tf.reduce_sum(self._V * weights, axis=0) / weights.sum(), (1, -1))

    def _step(self):
        V_alpha = self._compute_consensus_point()
        noise = self._noise.sample(self._V.shape)
        diff = self._V - V_alpha
        noise_weight = tf.abs(diff) if self._anisotropic else tf.reshape(tf.norm(diff, ord=2, axis=1), (-1, 1))
        self._V -= self._lambda * diff * self._dt
        self._V += self._sigma * noise * noise_weight * self._dt ** 0.5

    def _resource_apply_dense(self, grad, var, apply_state=None):
        self._step()
        var.assign(self._compute_consensus_point())

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        self._step()
        var.assign(self._compute_consensus_point())

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
