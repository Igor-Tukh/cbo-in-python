import torch


def cbo_update(V, V_alpha, anisotropic, l, sigma, dt, device=None):
    device = torch.device('cpu') if device is None else device
    noise = torch.randn(V.shape).to(device)
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
