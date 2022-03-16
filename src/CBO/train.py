import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm.auto import tqdm
from src.CBO.config import DEFAULT_OPTIMIZER_CONFIG, DEFAULT_INITIAL_DISTRIBUTION
from src.CBO.consensus_based_optimizer import CBO

tfd = tfp.distributions


class TensorboardLogging:
    def __init__(self, model_name, log_dir):
        self.model_name = model_name
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, model_name, 'train'))
        self.test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, model_name, 'test'))
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    def flush(self, epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)
        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)


class NeuralNetworkObjectiveFunction:
    def __init__(self, model, loss, X, y):
        self._model = model
        self._loss = loss
        self._X = X
        self._y = y

    # def _get_parameters(self):
    #     parameters = []
    #     for weight in self._model.trainable_weights:
    #         parameters.append(tf.reshape(weight, -1))
    #     return tf.concat(parameters, 0)

    def _substitute_parameters(self, parameters):
        update_model_parameters(self._model, parameters)

    def __call__(self, parameters):
        self._substitute_parameters(parameters)
        output = self._model(self._X)
        loss = self._loss(self._y, output)
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
          dataset_batches=None, X_val=None, y_val=None, tensorboard_logging=None, cooling=False):
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
    epoch = 0
    loss_model = tf.keras.models.clone_model(model)
    while np.less(timestamp, time_horizon):
        batches = np.array_split(np.random.permutation(X.shape[0]), dataset_batches)
        losses = []
        for i, batch in tqdm(enumerate(batches)):
            # TODO(itukh) we can pass the actual gradients to the `minimize` call bellow
            # with tf.GradientTape() as tape:
            #     logits = model(X[batch])
            #     loss_value = loss(y[batch], logits)
            # grads = tape.gradient(loss_value, model.trainable_weights)
            objective = NeuralNetworkObjectiveFunction(loss_model, loss, X[batch], y[batch])
            optimizer.update_objective(objective)
            # objective_function = lambda: objective(var)
            # grad_loss=tf.expand_dims(tf.zeros_like(var), 0)
            # optimizer.minimize(objective_function, [var], [tf.zeros_like(var)])
            optimizer.apply_gradients([(tf.zeros_like(var), var)])
            model = update_model_parameters(model, var)

            logits = model(X[batch])
            loss_value = loss(y[batch], logits)
            y_pred = tf.nn.softmax(logits)
            tensorboard_logging.train_loss(loss_value)
            tensorboard_logging.train_accuracy(y[batch], y_pred)

            accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            accuracy.update_state(y, model.predict(X))
            acc = accuracy.result().numpy()

            if X_val is not None:
                val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                val_accuracy.update_state(y_val, model.predict(X_val))
                val_acc = val_accuracy.result().numpy()

            if verbose:
                log = f'Epoch {epoch}, batch {i + 1}/{len(batches)}, ' \
                      f'batch objective: {objective(var)}, ' \
                      f'train accuracy: {acc}'
                if X_val is not None:
                    log += f', val accuracy: {val_acc}'
                print(log, end='\r')

            if return_trajectory:
                trajectory[timestamp] = {
                    'consensus': optimizer.minimizer(),
                    'particles': optimizer.particles(),
                    'batch': batch,
                    'accuracy': acc,
                    'var': var.numpy().copy(),
                }

            losses.append(objective(var))
            timestamp += optimizer_config['dt']

        if tensorboard_logging is not None:
            if X_val is not None:
                logits = model(X_val)
                loss_value = loss(y_val, logits)
                y_pred = tf.nn.softmax(logits)
                tensorboard_logging.test_loss(loss_value)
                tensorboard_logging.test_accuracy(y_val, y_pred)
            tensorboard_logging.flush(epoch)
        epoch += 1
        if cooling:
            optimizer.apply_cooling(epoch)
    update_model_parameters(model, var)
    if return_trajectory:
        return model, trajectory
    return model
