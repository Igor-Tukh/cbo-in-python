import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.extend([os.pardir, os.path.join(os.pardir, os.pardir)])

from src.CBO.consensus_based_optimizer import CBO
from tqdm import tqdm

tfd = tfp.distributions

DEFAULT_OPTIMIZER_CONFIG = {
    'dt': 0.02,
    'lmbda': 1,
    'sigma': 1,
    'alpha': 100,
    'anisotropic': True,
}


def minimize(objective, dimensionality, n_particles, time_horizon, optimizer_config=None, initial_distribution=None,
             return_trajectory=False, verbose=True):
    if optimizer_config is None:
        optimizer_config = DEFAULT_OPTIMIZER_CONFIG.copy()
    else:
        current_config = DEFAULT_OPTIMIZER_CONFIG.copy()
        current_config.update(optimizer_config)
        optimizer_config = current_config
    if initial_distribution is None:
        initial_distribution = tfd.Normal(1, 4)
    optimizer_config.update({
        'initial_particles': initial_distribution.sample((n_particles, dimensionality)),
        'objective': objective,
    })
    optimizer = CBO(**optimizer_config)
    trajectory = {}
    var = tf.Variable([initial_distribution.sample(dimensionality)])
    range_wrapper = tqdm if verbose else lambda i: i
    for timestamp in range_wrapper(np.arange(0, time_horizon + 1e-9, optimizer_config['dt'])):
        loss = lambda: objective(var)
        optimizer.minimize(loss, [var])
        if return_trajectory:
            trajectory[timestamp] = {
                'consensus': optimizer.minimizer(),
                'particles': optimizer.particles(),
            }
    if return_trajectory:
        return var.value(), trajectory
    return tf.reshape(var.value(), (-1,))


def minimize_for_ranges(objective, dimensionality, time_horizon, default_optimizer_config,
                        initial_distribution, alphas=None, ns_particles=None,
                        sigmas=None, dts=None, n_repeats=10, n_particles=100, verbose=False):
    def _generate_config(args_ranges_dict):
        def __generate_config(configs, current_config, keys, key_index):
            if key_index == len(keys):
                configs.append(current_config.copy())
                return configs
            current_key = keys[key_index]
            for value in args_ranges_dict[current_key]:
                current_config[current_key] = value
                configs = __generate_config(configs, current_config, keys, key_index + 1)
            return configs

        return __generate_config([], {}, list(args_ranges_dict.keys()), 0)

    ranges = {}
    for arg_name, arg_range in [('alpha', alphas), ('n_particles', ns_particles),
                                ('sigma', sigmas), ('dt', dts)]:
        if arg_range is not None:
            ranges[arg_name] = arg_range
    results = []
    for config in _generate_config(ranges):
        full_config = default_optimizer_config.copy()
        full_config.update(config)
        minimization_config = full_config.copy()
        if 'n_particles' in config:
            n_particles = config['n_particles']
            del minimization_config['n_particles']
        results.append({
            'config': full_config,
            'minimizers': [minimize(objective, dimensionality, n_particles, time_horizon,
                                    minimization_config, initial_distribution, verbose=verbose) for _ in range(n_repeats)],
        })
    return results
