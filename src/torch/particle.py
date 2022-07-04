import torch
import torch.nn as nn

import numpy as np

from copy import deepcopy


class Particle(nn.Module):
    def __init__(self, model):
        """
        Represents a particles in the consensus-based optimization. Stores a copy of the optimized model.
        :param model: the underlying model.
        """
        super(Particle, self).__init__()
        self.model = deepcopy(model)
        for p in self.model.parameters():
            with torch.no_grad():
                p.copy_(torch.randn_like(p))

    def forward(self, X):
        return self.model(X)

    def get_params(self):
        """
        :return: the underlying models' parameters stacked into a 1d-tensor.
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()]).view(-1)

    def set_params(self, new_params):
        """
        Updates the underlying models' parameters.
        :param new_params: new params stacked into a 1d-tensor.
        """
        next_slice = 0
        for p in self.model.parameters():
            slice_length = len(p.view(-1))
            with torch.no_grad():
                p.copy_(new_params[next_slice: next_slice + slice_length].view(p.shape))
            next_slice += slice_length

    def get_gradient(self):
        """
        Returns the gradients stacked into a 1d-tensor.
        """
        gradients = [p.grad for p in self.model.parameters()]
        if None in gradients:
            return None
        return torch.cat([g.view(-1) for g in gradients]).view(-1)
