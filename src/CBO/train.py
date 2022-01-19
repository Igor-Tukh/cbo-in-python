import tensorflow_probability as tfp

from src.CBO.consensus_based_optimizer import CBO

tfd = tfp.distributions


def train(X, y, loss, model, dimensionality, n_particles, time_horizon, optimizer_config=None,
          initial_distribution=None, return_trajectory=False, verbose=True, particles_batches=None,
          dataset_batches=None, validation=None):
    pass
