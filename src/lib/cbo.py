import numpy as np
import torch

import src.torch.cbo as torch_cbo

SUPPORTED_BACKENDS = {
    'torch',
}
SUPPORTED_BACKENDS_STR = ', '.join(SUPPORTED_BACKENDS)


def _determine_backend(*args):
    is_torch = np.all([isinstance(arg, torch.Tensor) for arg in args])
    # is_tensorflow = ...
    if is_torch:
        return 'torch'
    return None


def _determine_and_validate_backend(backend, *args):
    backend = _determine_backend(*args) if backend is None else backend
    if backend is None:
        raise RuntimeError(f'Cannot automatically identify backend to use. Please provide specify backend'
                           f'explicitly with the `backend` option. Supported backends: {SUPPORTED_BACKENDS_STR},')
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimeError(f'Provided backend option {backend} is not (currently) supported. '
                           f'Supported backends: {SUPPORTED_BACKENDS_STR}.')
    return backend


def cbo_update(V, V_alpha, anisotropic, l, sigma, dt, device, backend=None):
    backend = _determine_and_validate_backend(backend, V, V_alpha)
    if backend == 'torch':
        return torch_cbo.cbo_update(V, V_alpha, anisotropic, l, sigma, dt, device)


def compute_v_alpha(function_values, particles, alpha, backend=None):
    backend = _determine_and_validate_backend(backend, particles)
    if backend == 'torch':
        return torch_cbo.compute_v_alpha(function_values, particles, alpha)


def minimize(function, dimensionality, n_particles, n_particles_batches, time_horizon,
             initial_distribution, dt, l, sigma, alpha, anisotropic,
             use_multiprocessing, return_trajectory=False, backend='torch'):
    backend = _determine_and_validate_backend(backend)
    if backend == 'torch':
        pass
