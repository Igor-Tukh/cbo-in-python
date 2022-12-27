import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.getcwd().split('cbo-in-python')[0], 'cbo-in-python'))

from src.torch.cbo import minimize
from src.torch.standard_functions import rastrigin_c
from torch.distributions import Normal

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dim', type=int, default=1, help='optimization space dimensionality')
    parser.add_argument('--rastrigin_constant', type=float, default=2.5, help='constant for rastrigin function')

    parser.add_argument('--loc', type=float, default=0., help='initial distribution loc')
    parser.add_argument('--scale', type=float, default=1., help='initial distribution scale')

    parser.add_argument('--particles', type=int, default=100, help='use PARTICLES particles')
    parser.add_argument('--particles_batches', type=int, default=1, help='use PARTICLES_BATCHES particles batches')
    parser.add_argument('--epochs', type=int, default=10, help='minimize for EPOCHS epochs')

    parser.add_argument('--alpha', type=float, default=1000, help='alpha from CBO dynamics')
    parser.add_argument('--sigma', type=float, default=0.4 ** 0.5, help='sigma from CBO dynamics')
    parser.add_argument('--l', type=float, default=1, help='lambda from CBO dynamics')
    parser.add_argument('--dt', type=float, default=0.1, help='dt from CBO dynamics')
    parser.add_argument('--anisotropic', type=bool, default=True, help='whether to use anisotropic or not')
    parser.add_argument('--eps', type=float, default=1e-5, help='threshold for additional random shift')
    parser.add_argument('--partial_update', type=bool, default=True, help='whether to use partial or full update')

    args = parser.parse_args()

    objective = rastrigin_c(args.rastrigin_constant)
    print(f'Minimizing rastrigin function with a constant {args.rastrigin_constant}')
    start_time = time.time()
    minimizer = minimize(
        function=objective, dimensionality=args.dim, n_particles=args.particles,
        initial_distribution=Normal(args.loc, args.scale),
        n_particles_batches=args.particles_batches, dt=args.dt, l=args.l, sigma=args.sigma, alpha=args.alpha,
        anisotropic=args.anisotropic, epochs=args.epochs, return_trajectory=False)
    elapsed_time = time.time() - start_time
    print(f'Finished after {args.epochs} and {elapsed_time} seconds. Final minimizer: {minimizer}.')
