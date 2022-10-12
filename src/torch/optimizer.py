# TODO(itukh): fix energy values calculation

import multiprocessing

import torch
import numpy as np

from src.torch.particle import Particle
from src.torch.cbo import cbo_update, compute_v_alpha
from src.torch.utils import inplace_randn

from torch.utils.data import DataLoader


class Optimizer:
    def __init__(self, model, n_particles=10, l=1, alpha=100, sigma=1, dt=0.01, anisotropic=True, eps=1e-2,
                 use_multiprocessing=False, n_processes=4, particles_batch_size=None, apply_random_drift=True,
                 gamma=None, device=None, partial_update=False, apply_common_drift=False,
                 evaluation_strategy='last'):
        """
        Consensus based optimizer.
        :param model: model to optimize.
        :param n_particles: number of particles to use in the optimization.
        :param l: alias for `lambda`, CBO hyperparameter.
        :param alpha: CBO hyperparameter.
        :param sigma: CBO hyperparameter.
        :param dt: CBO dynamics time step.
        :param anisotropic: boolean flag indicating whether to use the anisotropic noise or not.
        :param eps: argument indicating how small the consensus update has to be to apply the additional random shift
        (drift) to particles.
        :param use_multiprocessing: whether to use multiprocessing where possible.
        :param n_processes: number of processes to use for multiprocessing.
        :param particles_batch_size: batch size for particle-level batching. If not specified, no batching will be used.
        :param apply_random_drift: whether to apply additional random shift (drift) or not.
        :param gamma: coefficient of a gradient drift. If gamma is None, no gradient drift will be applied.
        :param partial_update: whether to apply CBO update to all particles, or just to the corresponding bunch of
        particles.
        """
        # CBO hyperparameters
        self.n_particles = n_particles
        self.l = l
        self.alpha = alpha
        self.sigma = sigma
        self.dt = dt
        self.anisotropic = anisotropic
        # CBO additional hyperparameters
        self.gamma = gamma
        self.eps = eps  # specifies how small the consensus updated has to be to apply the additional shift
        # CBO dynamics additional settings
        self.apply_random_drift = apply_random_drift
        self.apply_common_drift = apply_common_drift
        self.partial_update = partial_update
        if evaluation_strategy not in {'last', 'full', 'best'}:
            raise ValueError(f'Unknown model evaluation strategy: {evaluation_strategy}')
        self.evaluation_strategy = evaluation_strategy
        # Multiprocessing settings
        self.use_multiprocessing = use_multiprocessing
        self.n_processes = min(n_processes, multiprocessing.cpu_count())
        # Device (CPU / GPU / TPU) settings
        self.device = torch.device('cpu') if device is None else device
        if self.use_multiprocessing and self.device.type == 'cuda':
            raise RuntimeError('Unable to use multiprocessing along with cuda')
        # Initialize required internal fields
        self.time = 0
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
        self.energy_values = None
        # Initialize particles
        self.model = model
        self._initialize_particles()
        self.particles_batch_size = particles_batch_size if particles_batch_size is not None else self.n_particles
        self.particles_dataloader = DataLoader(np.arange(self.n_particles), batch_size=self.particles_batch_size,
                                               shuffle=True)
        if device is not None:
            self.to(device)
        # Constants
        self.infinity = 1e5

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
        self.X = X.to(self.device)
        self.y = y.to(self.device)
        self.outputs = self._compute_particles_outputs()

    def compute_consensus(self, batch=None, alpha=None):
        """
        Returns the consensus computed based on the current particles positions.
        """
        outputs = self.outputs if batch is None else [self.outputs[i] for i in batch]
        # TODO(itukh): check this line
        values = torch.FloatTensor([self.loss(output, self.y) for output in outputs])
        alpha = self.alpha if alpha is None else alpha
        return compute_v_alpha(values, self.V[batch], alpha, self.device)

    def step(self):
        """
        Execute one step of the CBO dynamics.
        """
        if self.X is None:
            raise RuntimeError('Unable to perform the step without the prior loss.backward() call')
        self.V = self._get_particles_params()
        # TODO: parallelize the line below?
        self.energy_values = torch.FloatTensor([self.loss(output, self.y) for output in self.outputs]).to(self.device)
        if self.use_multiprocessing:
            self.V_alpha_old = self.V_alpha.clone() if self.V_alpha is not None else None
            self.V_alpha = self.compute_consensus(batch=self._generate_random_batch())  # TODO: check this line

            batches = [batch for batch in self.particles_dataloader]
            params = [(self.energy_values[batch].detach(), self.V[batch].detach(), self.alpha, self.anisotropic,
                       self.l, self.sigma, self.dt) for batch in batches]
            with multiprocessing.Pool(processes=self.n_processes) as pool:
                new_V = pool.starmap(_batch_step, params)
            for new_batch_V, batch in zip(new_V, batches):
                self.V[batch] = new_batch_V
            self._maybe_apply_random_shift()
            self._maybe_apply_gradient_shift()
        else:
            for particles_batch in self.particles_dataloader:
                self.V_alpha_old = self.V_alpha.clone() if self.V_alpha is not None else None
                self.V_alpha = compute_v_alpha(self.energy_values[particles_batch], self.V[particles_batch], self.alpha,
                                               self.device)
                if self.partial_update:
                    self.V[particles_batch] = cbo_update(self.V[particles_batch], self.V_alpha, self.anisotropic,
                                                         self.l, self.sigma, self.dt, self.device)
                else:
                    self.V = cbo_update(self.V, self.V_alpha, self.anisotropic,
                                        self.l, self.sigma, self.dt, self.device)
                self._maybe_apply_random_shift()
                self._maybe_apply_gradient_shift(batch=particles_batch)
        self._set_particles_params(self.V)
        self._maybe_apply_common_drift()
        self._update_model_params()
        self.time += self.dt

    def backward(self, loss):
        """
        Applies backpropagation for each dynamics particle.
        """
        if self.outputs is None:
            self.outputs = self._compute_particles_outputs()
        for output in self.outputs:
            loss_value = loss(output, self.y)
            loss_value.backward()

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

    def _generate_random_batch(self):
        return np.random.choice(np.arange(self.n_particles), self.particles_batch_size, replace=False)

    def to(self, device):
        """
        Transfers optimization to a new device. Typical application is cuda usage.
        """
        self.device = device
        for i, particle in enumerate(self.particles):
            self.particles[i] = particle.to(device)
        self.model = self.model.to(device)
        if self.X is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def _maybe_apply_random_shift(self):
        if not self.apply_random_drift:
            return
        if self.V_alpha_old is not None:
            norm = torch.norm(self.V_alpha.view(-1) - self.V_alpha_old.view(-1), p=float('inf'),
                              dim=0).detach().cpu().numpy()
            if np.less(norm, self.eps):
                self.V += self.sigma * (self.dt ** 0.5) * inplace_randn(self.V.shape, self.device)

            self.shift_norm = norm

    def _maybe_apply_gradient_shift(self, batch=None):
        if self.gamma is None:
            return
        batch = np.arange(self.V.shape[0]) if batch is None else batch
        self.V[batch] -= torch.cat([self.particles[i].get_gradient() for i in batch]).view(
            self.V[batch].shape) * self.gamma

    def _maybe_apply_common_drift(self):
        if not self.apply_common_drift:
            return
        self.outputs = self._compute_particles_outputs()
        self.energy_values = torch.FloatTensor([self.loss(output, self.y) for output in self.outputs]).to(self.device)
        self.V_alpha = compute_v_alpha(self.energy_values, self.V, self.alpha, self.device)
        self.V = cbo_update(self.V, self.V_alpha, self.anisotropic, self.l, self.sigma, self.dt, self.device)
        self._set_particles_params(self.V)

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
            with multiprocessing.Pool(processes=self.n_processes) as pool:
                values = pool.starmap(_forward, [(p, self.X) for p in self.particles])
        else:
            for particle in self.particles:
                values.append(particle(self.X))
        return values

    def _update_model_params(self):
        new_params = None
        if self.evaluation_strategy == 'last':
            new_params = self.V_alpha
        elif self.evaluation_strategy == 'full':
            self.V_alpha = self.compute_consensus()
            new_params = self.V_alpha
        elif self.evaluation_strategy == 'best':
            self.V_alpha = self.compute_consensus(alpha=self.infinity)
            new_params = self.V_alpha
        if new_params is None:
            return
        next_slice = 0
        for p in self.model.parameters():
            slice_length = len(p.view(-1))
            with torch.no_grad():
                p.copy_(new_params[next_slice: next_slice + slice_length].view(p.shape))
            next_slice += slice_length


# Multiprocessing helper functions

def _forward(model, X):
    return model(X).detach()


def _batch_step(energy_values, V, alpha, anisotropic, l, sigma, dt):
    V_alpha = compute_v_alpha(energy_values, V, alpha, self.device)
    return cbo_update(V, V_alpha, anisotropic, l, sigma, dt)
