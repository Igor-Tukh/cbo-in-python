import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tqdm import tqdm


def visualize_trajectory_1d(trajectory, objective, output_path=None, x_range=None):
    if output_path is None:
        output_path = 'trajectory.gif'
    if x_range is None:
        x_range = (-10, 10)
    points = np.linspace(*x_range, 1000)
    values = [objective(tf.constant(point, dtype=tf.float32)) for point in points]
    plt.rcParams.update({'figure.figsize': (10, 10)})
    plt.ioff()
    with imageio.get_writer(output_path) as writer:
        for i, timestamp in enumerate(tqdm(sorted(list(trajectory.keys())))):
            point = trajectory[timestamp]
            plt.clf()
            plt.title('t = %.2f' % round(timestamp, 3))
            plt.xlim([points.min(), points.max()])
            plt.ylim([np.min(values), np.max(values)])
            plt.plot(points, values, color='orange')
            for particle in point['particles']:
                plt.plot([particle], [objective(particle)], marker='o', color='black')
            consensus_point = tf.reshape(point['consensus'], (-1,))
            plt.plot([consensus_point], [objective(consensus_point)], marker='o', color='r', markersize=10)
            plot_path = os.path.join(os.path.dirname(output_path), f'{i}_tmp.png')
            plt.savefig(plot_path)
            image = imageio.imread(plot_path)
            writer.append_data(image)
            os.remove(plot_path)


def visualize_trajectory_convergence(trajectory, minimizer, display_exponent=False, exp_lambda=1., exp_sigma=1.,
                                     output_path=None):
    plt.rcParams.update({'figure.figsize': (10, 10)})
    convergence_measure = lambda points: tf.reduce_sum((tf.norm(points - minimizer, ord=2, axis=1) ** 2)) / 2
    timestamps = np.array(list(sorted(trajectory.keys())))
    values = np.array([convergence_measure(trajectory[t]['particles']) for t in timestamps])
    plt.clf()
    plt.plot(timestamps, values / values[0], label='result')
    if display_exponent:
        plt.plot(timestamps, np.exp(-timestamps * (2 * exp_lambda - exp_sigma ** 2)),
                 label=r'$e^{-(2\lambda - \sigma^2)t}$', linestyle='--', alpha=0.5)
        plt.legend(prop={'size': 18})
    plt.xlabel('t', fontsize=18)
    plt.ylabel(r'$\frac{V^N(\rho_t^N)}{V^N(\rho_0^N)}$', fontsize=18)
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()


def visualize_minimization_successfulness(x_name, y_name, minimization_results, success_criterion, output_path=None):
    def _process_minimization_results():
        x_values = []
        y_values = []
        success_rates = []
        for r in minimization_results:
            x_values.append(r['config'][x_name])
            y_values.append(r['config'][y_name])
            success_rates.append(np.array([success_criterion(m) for m in r['minimizers']], dtype=np.float).mean())
        return pd.DataFrame({x_name: x_values, y_name: y_values, 'mean success': success_rates}).pivot(x_name, y_name,
                                                                                                       'mean success')

    plt.rcParams.update({'figure.figsize': (10, 10),
                         'font.size': 18})
    _, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(_process_minimization_results(), cmap='YlGnBu', vmin=0, vmax=1)
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()
