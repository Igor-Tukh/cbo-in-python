import torch


def cbo_update(V, V_alpha, anisotropic, l, sigma, dt):
    noise = torch.randn(V.shape)
    diff = V - V_alpha
    noise_weight = torch.abs(diff) if anisotropic else torch.norm(diff, p=2, dim=1)
    V -= l * diff * dt
    V += sigma * noise_weight * noise * (dt ** 0.5)
    return V


def compute_v_alpha(energy_values, particles, alpha):
    weights = torch.exp(-alpha * (energy_values - energy_values.min())).reshape(-1, 1)
    consensus = (weights * particles) / weights.sum()
    return consensus.sum(dim=0)
