import torch

import src.torch.cbo as torch_cbo


def cbo_update(V, V_alpha, anisotropic, l, sigma, dt):
    if isinstance(V, torch.Tensor) and isinstance(V_alpha, torch.Tensor):
        return torch_cbo.cbo_update(V, V_alpha, anisotropic, l, sigma, dt)
    raise NotImplementedError(f'Provided particles (V) type: {type(V)} is not currently supported.')


def compute_v_alpha(function_values, particles, alpha):
    if isinstance(particles, torch.Tensor):
        return torch_cbo.compute_v_alpha(function_values, particles, alpha)
