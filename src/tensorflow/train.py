import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import wandb


from tqdm.auto import tqdm
from src.tensorflow.config import DEFAULT_OPTIMIZER_CONFIG, DEFAULT_INITIAL_DISTRIBUTION
from src.tensorflow.consensus_based_optimizer import CBO

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


class UpdatableTfModel:
    def __init__(self, model):
        self._model = tf.keras.models.clone_model(model)
        self._is_trainable = self._get_trainable_weights_mask()

    def get_weights(self):
        return np.concatenate([weight.reshape(-1) for weight in self._model.get_weights()]).reshape(-1)

    def get_model(self):
        return self._model

    def _get_trainable_weights_mask(self):
        trainable_weights_names = set([w.name for w in self._model.trainable_weights])
        weights_names = [w.name for w in self._model.weights]
        return np.vectorize(lambda n: n in trainable_weights_names)(weights_names)

    def set_weights(self, weights):
        if isinstance(weights, tf.Tensor) or isinstance(weights, tf.Variable):
            weights = weights.numpy()
        weights = weights.reshape(-1)
        new_weights = []
        current_position = 0
        for i, weight in enumerate(self._model.get_weights()):
            if self._is_trainable[i]:
                weight_len = weight.reshape(-1).shape[0]
                new_weights.append(weights[current_position:current_position + weight_len].reshape(weight.shape))
                current_position += weight_len
            else:
                new_weights.append(weight)
        self._model.set_weights(new_weights)


class NeuralNetworkObjectiveFunction:
    def __init__(self, model, loss, X, y):
        self._model = UpdatableTfModel(model)
        self._loss = loss
        self._X = X
        self._y = y

    def __call__(self, parameters, training=True):
        self._model.set_weights(parameters)
        output = self._model.get_model()(self._X, training=training)
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
          update_all_particles=True, dataset_batches=None, X_val=None, y_val=None, tensorboard_logging=None,
          cooling=False, evaluation_sample_size=None, evaluation_rate=None, use_multiprocessing=False, eps=1e-3):
    dimensionality = compute_model_dimensionality(model)
    if optimizer_config is None:
        optimizer_config = DEFAULT_OPTIMIZER_CONFIG.copy()
    else:
        current_config = DEFAULT_OPTIMIZER_CONFIG.copy()
        current_config.update(optimizer_config)
        optimizer_config = current_config
    if initial_distribution is None:
        initial_distribution = DEFAULT_INITIAL_DISTRIBUTION

    loss_model = tf.keras.models.clone_model(model)
    # overall_objective = NeuralNetworkObjectiveFunction(loss_model, loss, X, y)

    optimizer_config.update({
        'initial_particles': initial_distribution.sample((n_particles, dimensionality)),
        'objective': None,
        'n_batches': particles_batches,
        'update_all_particles': update_all_particles,
        'use_multiprocessing': use_multiprocessing,
        'eps': eps,
    })
    optimizer = CBO(**optimizer_config)

    dataset_batches = 1 if dataset_batches is None else dataset_batches
    trajectory = {}
    var = tf.Variable(initial_distribution.sample(dimensionality))
    timestamp = 0
    epoch = 0
    model = UpdatableTfModel(model)

    while np.less(timestamp, time_horizon):
        batches = np.array_split(np.random.permutation(X.shape[0]), dataset_batches)
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

            if evaluation_rate is None or i % evaluation_rate == 0:
                training_subset = np.arange(X.shape[0])
                training_subset = training_subset if evaluation_sample_size is None else np.random.choice(training_subset,
                                                                                                          evaluation_sample_size,
                                                                                                          replace=False)

                approximate_objective = NeuralNetworkObjectiveFunction(loss_model, loss, X[training_subset],
                                                                       y[training_subset])
                approximate_v_alpha = optimizer.minimizer(objective=approximate_objective)
                model.set_weights(approximate_v_alpha)

                logits = model.get_model()(X[training_subset], training=True)
                loss_value = loss(y[training_subset], logits)
                y_pred = tf.nn.softmax(logits)

                if tensorboard_logging is not None:
                    tensorboard_logging.train_loss(loss_value)
                    tensorboard_logging.train_accuracy(y[training_subset], y_pred)

                accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                accuracy.update_state(y, model.get_model()(X, training=True))
                acc = accuracy.result().numpy()

                if X_val is not None:
                    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                    val_accuracy.update_state(y_val, model.get_model()(X_val, training=False))
                    val_acc = val_accuracy.result().numpy()

                if verbose:
                    obj = objective(var).numpy()
                    wandb.log({
                        'epoch': epoch,
                        'train_accuracy': acc,
                        'train_loss': obj,
                        'val_accuracy': val_acc,
                        'consensus_shift': optimizer.consensus_shift,
                    })
                    #  # f'objective: {str(round(overall_objective(var).numpy(), 3))}, ' \
                    log = f'Epoch {epoch}, batch {i + 1}/{len(batches)}, ' \
                          f'batch objective: {str(round(obj, 3))}, ' \
                          f'train accuracy: {str(round(acc, 3))}'
                    if X_val is not None:
                        log += f', val accuracy: {str(round(val_acc, 3))}'
                    print(log, end='\r')

                if return_trajectory:
                    trajectory[timestamp] = {
                        'consensus': optimizer.minimizer(),
                        'particles': optimizer.particles(),
                        'batch': batch,
                        'accuracy': acc,
                        'var': var.numpy().copy(),
                    }

            # losses.append(overall_objective(var))
            timestamp += optimizer_config['dt']

        if tensorboard_logging is not None:
            if X_val is not None:
                logits = model.get_model()(X_val, training=False)
                loss_value = loss(y_val, logits)
                y_pred = tf.nn.softmax(logits)
                tensorboard_logging.test_loss(loss_value)
                tensorboard_logging.test_accuracy(y_val, y_pred)
            tensorboard_logging.flush(epoch)
        epoch += 1
        if cooling:
            optimizer.apply_cooling(epoch)

    full_data_objective = NeuralNetworkObjectiveFunction(loss_model, loss, X, y)
    var.assign(tf.reshape(optimizer.minimizer(objective=full_data_objective), -1))
    model.set_weights(var)

    if return_trajectory:
        return model.get_model(), trajectory
    return model.get_model()
