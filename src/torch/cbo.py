import torch
import numpy as np

from src.torch.utils import inplace_randn
from torch.utils.data import DataLoader


def cbo_update(V, V_alpha, anisotropic, l, sigma, dt, device=None):
    device = torch.device('cpu') if device is None else device
    noise = inplace_randn(V.shape, device)
    with torch.no_grad():
        diff = V - V_alpha
        noise_weight = torch.abs(diff) if anisotropic else torch.norm(diff, p=2, dim=1)
        V -= l * diff * dt
        V += sigma * noise_weight * noise * (dt ** 0.5)
    return V


def compute_v_alpha(energy_values, particles, alpha, device=None):
    device = torch.device('cpu') if device is None else device
    weights = torch.exp(-alpha * (energy_values - energy_values.min())).reshape(-1, 1).to(device)
    consensus = (weights * particles) / weights.sum()
    return consensus.sum(dim=0)


def compute_energy_values(function, V, device=None):
    # TODO(itukh) is it possible to apply vectorization here to improve the performance
    device = torch.device('cpu') if device is None else device
    return torch.stack([function(v) for v in V]).to(device)


def minimize(
        # General CBO / optimization parameters
        function, dimensionality, n_particles,
        initial_distribution, dt, l, sigma, alpha, anisotropic,
        # Optimization parameters
        batch_size=None, n_particles_batches=None,
        epochs=None, time_horizon=None,
        # Optimization modifications parameters
        use_partial_update=False,
        use_additional_random_shift=False,
        use_additional_gradients_shift=False,
        random_shift_epsilon=None,
        gradients_shift_gamma=None,
        # Additional optional arguments
        best_particle_alpha=1e5,
        use_gpu_if_available=False,
        use_multiprocessing=False,
        return_trajectory=False):
    # Setting up computations on GPU / CPU
    device = torch.device('cuda') if (use_gpu_if_available and torch.cuda.is_available()) else torch.device('cpu')
    # Standardize input arguments
    batch_size = int(n_particles // n_particles_batches) if batch_size is None else batch_size
    epochs = int(time_horizon // dt) if epochs is None else epochs
    # Initialize variables
    V = initial_distribution.sample((n_particles, dimensionality)).to(device)
    if use_additional_gradients_shift:
        V.requires_grad = True
    V_batches = DataLoader(np.arange(n_particles), batch_size=batch_size, shuffle=True)
    V_alpha_old = None

    # Main optimization loop
    trajectory = []
    for epoch in range(epochs):
        for batch in V_batches:
            V_batch = V[batch]
            batch_energy_values = compute_energy_values(function, V_batch, device=device)
            V_alpha = compute_v_alpha(batch_energy_values, V_batch, alpha, device=device)

            if use_partial_update:
                V[batch] = cbo_update(V_batch, V_alpha, anisotropic, l, sigma, dt, device=device)
            else:
                V = cbo_update(V, V_alpha, anisotropic, l, sigma, dt, device=device)

            if use_additional_random_shift:
                if V_alpha_old is None:
                    continue
                norm = torch.norm(V_alpha.view(-1) - V_alpha_old.view(-1), p=float('inf'), dim=0).detach().cpu().numpy()
                if np.less(norm, random_shift_epsilon):
                    V += sigma * (dt ** 0.5) * inplace_randn(V.shape, device=device)
                V_alpha_old = V_alpha

        if use_additional_gradients_shift:
            if V.grad is not None:
                V.grad.zero_()
            energy_values = compute_energy_values(function, V, device=device)
            loss = energy_values.sum()
            loss.backward()
            with torch.no_grad():
                V -= gradients_shift_gamma * V.grad

        if return_trajectory:
            energy_values = compute_energy_values(function, V, device=device)
            V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)
            V_best = compute_v_alpha(energy_values, V, best_particle_alpha, device=device)
            trajectory.append(
                {
                    'V': V.clone().detach().cpu(),
                    'V_alpha': V_alpha.clone().detach().cpu(),
                    'V_best': V_best.clone().detach().cpu(),
                }
            )

    energy_values = compute_energy_values(function, V, device=device)
    V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)
    if return_trajectory:
        return V_alpha.detach().cpu(), trajectory
    return V_alpha.detach().cpu()
