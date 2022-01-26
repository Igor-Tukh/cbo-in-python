import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm.auto import tqdm
from src.CBO.config import DEFAULT_OPTIMIZER_CONFIG, DEFAULT_INITIAL_DISTRIBUTION
from src.CBO.consensus_based_optimizer import CBO

tfd = tfp.distributions


class NeuralNetworkObjectiveFunction:
    def __init__(self, model, loss, X, y):
        self._model = model
        self._loss = loss
        self._X = X
        self._y = y
        self._initial_parameters = self._get_parameters()

    def _get_parameters(self):
        parameters = []
        for weight in self._model.trainable_weights:
            parameters.append(tf.reshape(weight, -1))
        return tf.concat(parameters, 0)

    def _substitute_parameters(self, parameters):
        update_model_parameters(self._model, parameters)

    def __call__(self, parameters):
        self._substitute_parameters(parameters)
        output = self._model(self._X)
        loss = self._loss(self._y, output)
        self._substitute_parameters(self._initial_parameters)
        return loss


def compute_model_dimensionality(model):
    return np.sum([tf.size(weight) for weight in model.trainable_weights])


def update_model_parameters(model, parameters):
    current_position = 0
    for weight in model.trainable_weights:
        next_position = current_position + tf.size(weight)
        weight.assign(tf.reshape(parameters[current_position:next_position], weight.shape))
        current_position = next_position
    return model


def train(model, loss, X, y, n_particles, time_horizon, optimizer_config=None,
          initial_distribution=None, return_trajectory=False, verbose=True, particles_batches=None,
          dataset_batches=None, X_val=None, y_val=None):
    dimensionality = compute_model_dimensionality(model)
    if optimizer_config is None:
        optimizer_config = DEFAULT_OPTIMIZER_CONFIG.copy()
    else:
        current_config = DEFAULT_OPTIMIZER_CONFIG.copy()
        current_config.update(optimizer_config)
        optimizer_config = current_config
    if initial_distribution is None:
        initial_distribution = DEFAULT_INITIAL_DISTRIBUTION
    optimizer_config.update({
        'initial_particles': initial_distribution.sample((n_particles, dimensionality)),
        'objective': None,
        'n_batches': particles_batches,
    })
    dataset_batches = 1 if dataset_batches is None else dataset_batches
    optimizer = CBO(**optimizer_config)
    trajectory = {}
    var = tf.Variable(initial_distribution.sample(dimensionality))
    timestamp = 0
    while np.less(timestamp, time_horizon):
        batches = np.array_split(np.random.permutation(X.shape[0]), dataset_batches)
        losses = []
        for batch in tqdm(batches):
            # TODO(itukh) we can pass the actual gradients to the `minimize` call bellow
            # with tf.GradientTape() as tape:
            #     logits = model(X[batch])
            #     loss_value = loss(y[batch], logits)
            # grads = tape.gradient(loss_value, model.trainable_weights)

            objective = NeuralNetworkObjectiveFunction(model, loss, X[batch], y[batch])
            optimizer.update_objective(objective)
            # objective_function = lambda: objective(var)
            # grad_loss=tf.expand_dims(tf.zeros_like(var), 0)
            # optimizer.minimize(objective_function, [var], [tf.zeros_like(var)])
            optimizer.apply_gradients([(tf.zeros_like(var), var)])
            if return_trajectory:
                trajectory[timestamp] = {
                    'consensus': optimizer.minimizer(),
                    'particles': optimizer.particles(),
                    'batch': batch,
                }
            losses.append(objective(var))
            timestamp += optimizer_config['dt']
        if verbose:
            print(f'Timestamp: {round(timestamp, 2)}, loss: {np.mean(losses)}', end='')
            if X_val is not None:
                val_loss = NeuralNetworkObjectiveFunction(model, loss, X_val, y_val)(var)
                print(f', validation loss: {val_loss}', end='')
            print()
    update_model_parameters(model, var)
    if return_trajectory:
        return model, trajectory
    return model
