import os

import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np

from tqdm import tqdm


def _set_default_rc_params():
    plt.rcParams.update({'figure.figsize': (15, 15)})
    plt.ioff()


def visualize_trajectory_1d(trajectory, function, dt, output_path=None, x_range=None):
    _set_default_rc_params()
    if output_path is None:
        output_path = 'trajectory.gif'

    if x_range is None:
        x_range = (-10, 10)
    xs = torch.linspace(*x_range, 1000)
    function_values = np.array([function(x) for x in xs])

    images = []

    for i, trajectory_point in enumerate(trajectory):
        V, V_alpha, V_best = trajectory_point['V'], trajectory_point['V_alpha'], trajectory_point['V_best']

        plt.clf()
        plt.title(f't = {(dt * i):.2f}, V_alpha = {V_alpha.item():.2f}')
        plt.ylim([function_values.min(), function_values.max()])
        plt.xlim([x_range[0], x_range[1]])

        plt.plot(xs, function_values, color='orange', alpha=0.5)
        plt.scatter(V, [function(v) for v in V], marker='o', color='black', label='particles')
        plt.scatter([V_best], [function(V_best)], marker='o', color='blue', label='best particle')
        plt.scatter([V_alpha], [function(V_alpha)], marker='o', color='red', label='consensus')

        plt.legend()

        plot_path = os.path.join(os.path.dirname(output_path), f'{i}_tmp.png')
        plt.savefig(plot_path)
        image = iio.imread(plot_path)
        images.append(image)
        os.remove(plot_path)

    plt.clf()
    if os.path.exists(output_path):
        # Is required for a correct gifs rendering in notebooks
        os.remove(output_path)
    imageio.mimsave(output_path, images)


def visualize_trajectory_convergence(trajectory, minimizer, display_exponent=False, l=1., sigma=1., dt=0.01,
                                     output_path=None):
    plt.rcParams.update({'figure.figsize': (10, 10)})
    convergence_measure = lambda points: (torch.norm(points - minimizer, p=2, dim=1) ** 2).sum().detach().numpy() / 2
    timestamps = dt * np.arange(len(trajectory))
    values = np.array([convergence_measure(t['V']) for t in trajectory])
    plt.clf()
    plt.plot(timestamps, values / values[0], label='result')
    if display_exponent:
        plt.plot(timestamps, np.exp(-timestamps * (2 * l - sigma ** 2)),
                 label=r'$e^{-(2\lambda - \sigma^2)t}$', linestyle='--', alpha=0.5)
        plt.legend(prop={'size': 18})
    plt.xlabel('t', fontsize=18)
    plt.ylabel(r'$\frac{V^N(\rho_t^N)}{V^N(\rho_0^N)}$', fontsize=18)
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()
