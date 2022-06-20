import multiprocessing

import torch
import numpy as np

from src.torch.particle import Particle
from src.torch.cbo import cbo_update, compute_v_alpha

from torch.utils.data import DataLoader


class Optimizer:
    # TODO: does it need to be inherited from `torch.optim.Optimizer`?
    def __init__(self, model, n_particles=10, l=1, alpha=100, sigma=1, dt=0.01, anisotropic=True, eps=1e-2,
                 use_multiprocessing=False, particles_batch_size=None):
        """
        Consensus based optimizer.
        :param model: model to optimize.
        :param n_particles: number of particles to use in the optimization.
        :param l: alias for `lambda`, CBO hyperparameter.
        :param alpha: CBO hyperparameter.
        :param sigma: CBO hyperparameter.
        :param dt: CBO dynamics time step.
        :param anisotropic: boolean flag indicating whether to use the anisotropic noise or not.
        :param eps: argument indicating how small the consensus update has to be to apply the additional random shift to
        particles.
        :param use_multiprocessing: whether to use multiprocessing where possible.
        :param particles_batch_size: batch size for particle-level batching. If not specified, no batching will be used.
        """
        # CBO hyperparameters
        self.n_particles = n_particles
        self.l = l
        self.alpha = alpha
        self.sigma = sigma
        self.dt = dt
        self.anisotropic = anisotropic
        # Additional hyperparameters
        self.eps = eps  # specifies how small the consensus updated has to be to apply the additional shift
        self.time = 0
        # Multiprocessing
        self.use_multiprocessing = use_multiprocessing
        self.pool = multiprocessing.Pool() if use_multiprocessing else None
        # Initialize required internal fields
        self.particles = []
        self.outputs = None
        self.loss = None
        self.X = None
        self.y = None
        self.V = None
        self.V_alpha = None
        self.V_alpha_old = None
        self.shift_norm = None
        self.consensus = None
        # Initialize particles
        self.model = model
        self._initialize_particles()
        self.particles_batch_size = particles_batch_size if particles_batch_size is not None else self.n_particles
        self.particles_dataloader = DataLoader(np.arange(self.n_particles), batch_size=self.particles_batch_size,
                                               shuffle=True, num_workers=min(2, multiprocessing.cpu_count()))

    def _initialize_particles(self):
        self.particles = [Particle(self.model) for _ in range(self.n_particles)]
        self.outputs = [None for _ in range(self.n_particles)]

    def _get_particles_params(self):
        return torch.stack([particle.get_params() for particle in self.particles])

    def _set_particles_params(self, new_particles_params):
        for particle, new_particle_params in zip(self.particles, new_particles_params):
            particle.set_params(new_particle_params)

    def _compute_particles_outputs(self):
        values = []
        if self.use_multiprocessing:
            values = self.pool.starmap(_forward, [(p, self.X) for p in self.particles])
        else:
            for particle in self.particles:
                values.append(particle(self.X))
        return torch.stack(values)

    def _update_model_params(self):
        if self.V_alpha is None:
            return
        next_slice = 0
        for p in self.model.parameters():
            slice_length = len(p.view(-1))
            with torch.no_grad():
                p.copy_(self.V_alpha[next_slice: next_slice + slice_length].view(p.shape))
            next_slice += slice_length

    def set_loss(self, loss):
        """
        Updates the optimization loss (energy) function.
        :param loss: new loss function. Should take as arguments the model outputs and targets respectively.
        """
        self.loss = loss

    def set_batch(self, X, y):
        """
        Updates the data batch to evaluate the energy function on.
        """
        self.X = X
        self.y = y
        self.outputs = self._compute_particles_outputs()

    def compute_consensus(self):
        """
        Returns the consensus computed based on the current particles positions.
        """
        # TODO: embed into CBO library, use the external call instead
        values = torch.FloatTensor([self.loss(output, self.y) for output in self.outputs])
        return compute_v_alpha(values, self.V, self.alpha)

    def step(self):
        """
        Execute one step of the CBO dynamics.
        """
        if self.X is None:
            raise RuntimeError('Unable to perform the step without the prior loss.backward() call')
        self.V = self._get_particles_params()
        for particles_batch in self.particles_dataloader:
            self.V_alpha_old = self.V_alpha.clone() if self.V_alpha is not None else None
            energy_values = torch.FloatTensor([self.loss(output, self.y) for output in self.outputs[particles_batch]])
            self.V_alpha = compute_v_alpha(energy_values, self.V[particles_batch], self.alpha)
            self.V = cbo_update(self.V, self.V_alpha, self.anisotropic, self.l, self.sigma, self.dt)
            self.maybe_apply_additional_shift()
        self._set_particles_params(self.V)
        self._update_model_params()
        self.time += self.dt

    def maybe_apply_additional_shift(self):
        if self.V_alpha_old is not None:
            norm = torch.norm(self.V_alpha.view(-1) - self.V_alpha_old.view(-1), p=float('inf'), dim=0).detach().numpy()
            if np.less(norm, self.eps):
                self.V += self.sigma * (self.dt ** 0.5) * torch.randn(self.V.shape)
            self.shift_norm = norm

    def zero_grad(self):
        """
        Zeroes the gradient values for all the particles and model. May be helpful when using the gradients.
        """
        for particle in self.particles:
            particle.zero_grad()
        self.model.zero_grad()

    def get_current_time(self):
        """
        Returns the current timestamp. Timestamp is incremented bt the `dt` on every optimization step,
        """
        return self.time


# Multiprocessing helper functions
def _forward(model, X):
    return model(X).detach()
