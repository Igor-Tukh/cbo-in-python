import os
import sys

sys.path.extend([os.pardir, os.path.join(os.pardir, os.pardir)])

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy

from src.torch import Optimizer
from src.torch import Loss
from src.datasets import load_mnist_dataloaders

EPOCHS = 1
BATCH_SIZE = 64
N_PARTICLES = 100
PARTICLES_BATCH_SIZE = 10
ALPHA = 50
LAMBDA = 1
SIGMA = 0.4 ** 0.5
DT = .1
ANISOTROPIC = True
EPS = 1e-5
USE_MULTIPROCESSING = True
N_PROCESSES = 6


def build_model():
    return nn.Sequential(
        nn.Flatten(1, 3),
        nn.Linear(28 ** 2, 10),
        nn.BatchNorm1d(10, affine=False),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.BatchNorm1d(10, affine=False),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.LogSoftmax(),
    )


def train():
    model = build_model()

    train_dataloader, test_dataloader = load_mnist_dataloaders(train_batch_size=BATCH_SIZE,
                                                               test_batch_size=BATCH_SIZE)

    optimizer = Optimizer(model, n_particles=N_PARTICLES, alpha=ALPHA, sigma=SIGMA,
                          l=LAMBDA, dt=DT, anisotropic=ANISOTROPIC, eps=EPS,
                          use_multiprocessing=USE_MULTIPROCESSING, n_processes=N_PROCESSES,
                          particles_batch_size=PARTICLES_BATCH_SIZE)

    loss_fn = Loss(F.nll_loss, optimizer)
    accuracy = Accuracy()
    n_batches = len(train_dataloader)

    def evaluate_model(X_, y_):
        with torch.no_grad():
            outputs = model(X_)
            y_pred = torch.argmax(outputs, dim=1)
        return loss_fn(outputs, y_), accuracy(y_pred.cpu(), y_.cpu())

    for epoch in range(EPOCHS):
        for batch, (X, y) in enumerate(train_dataloader):
            train_loss, train_acc = evaluate_model(X, y)
            optimizer.zero_grad()
            loss_fn.backward(X, y, backward_gradients=False)
            optimizer.step()

            with torch.no_grad():
                losses = []
                accuracies = []
                for X_test, y_test in test_dataloader:
                    loss, acc = evaluate_model(X_test, y_test)
                    losses.append(loss.cpu())
                    accuracies.append(acc.cpu())
                val_loss, val_acc = np.mean(losses), np.mean(accuracies)

            print(f'Epoch: {epoch + 1:2}/{EPOCHS}, batch: {batch + 1:4}/{n_batches}, '
                  f'train loss: {train_loss:8.3f}, train acc: {train_acc:8.3f}, '
                  f'val loss: {val_loss:8.3f}, val acc: {val_acc:8.3f}',
                  end='\r', flush=True)


if __name__ == '__main__':
    train()
