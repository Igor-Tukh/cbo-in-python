import os
import sys
import argparse
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd().split('cbo-in-python')[0], 'cbo-in-python'))

from src.torch.models import *
from src.datasets import load_mnist_dataloaders
from src.torch import Optimizer, Loss

MODELS = {
    'TinyMLP': TinyMLP,
    'SmallMLP': SmallMLP,
    'LeNet1': LeNet1,
    'LeNet5': LeNet5,
}

DATASETS = {
    'MNIST': load_mnist_dataloaders,
}


def _evaluate(model, X_, y_, loss_fn):
    with torch.no_grad():
        outputs = model(X_)
        y_pred = torch.argmax(outputs, dim=1)
        loss = loss_fn(outputs, y_)
        acc = 1. * y_.eq(y_pred).sum().item() / y_.shape[0]
    return loss, acc


def train(model, train_dataloader, test_dataloader, device, use_multiprocessing, processes,
          epochs, particles, particles_batch_size,
          alpha, sigma, l, dt, anisotropic, eps, partial_update, cooling,
          eval_freq):
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    optimizer = Optimizer(model, n_particles=particles, alpha=alpha, sigma=sigma,
                          l=l, dt=dt, anisotropic=anisotropic, eps=eps, partial_update=partial_update,
                          use_multiprocessing=use_multiprocessing, n_processes=processes,
                          particles_batch_size=particles_batch_size, device=device)
    loss_fn = Loss(F.nll_loss, optimizer)

    n_batches = len(train_dataloader)

    for epoch in range(epochs):
        epoch_train_accuracies = []
        epoch_train_losses = []
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            train_loss, train_acc = _evaluate(model, X, y, F.nll_loss)
            epoch_train_accuracies.append(train_acc)
            epoch_train_losses.append(train_loss.cpu())

            optimizer.zero_grad()
            loss_fn.backward(X, y, backward_gradients=False)
            optimizer.step()

            if batch % eval_freq == 0 or batch == n_batches - 1:
                with torch.no_grad():
                    losses = []
                    accuracies = []
                    for X_test, y_test in test_dataloader:
                        X_test, y_test = X_test.to(device), y_test.to(device)
                        loss, acc = _evaluate(model, X_test, y_test, F.nll_loss)
                        losses.append(loss.cpu())
                        accuracies.append(acc)
                    val_loss, val_acc = np.mean(losses), np.mean(accuracies)
                    if batch == n_batches - 1:
                        test_accuracies.append(val_acc)
                        test_losses.append(val_loss)

                print(
                    f'Epoch: {epoch + 1:2}/{epochs}, batch: {batch + 1:4}/{n_batches}, train loss: {train_loss:8.3f}, '
                    f'train acc: {train_acc:8.3f}, test loss: {val_loss:8.3f}, test acc: {val_acc:8.3f}',
                    flush=True)
        train_accuracies.append(np.mean(epoch_train_accuracies))
        train_losses.append(np.mean(epoch_train_losses))
        if cooling:
            optimizer.cooling_step()

    return train_accuracies, test_accuracies, train_losses, test_losses


def build_plot(epochs, model_name, dataset_name, plot_path,
               train_acc, test_acc, train_loss, test_loss):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 25

    epochs_range = np.arange(1, epochs + 1, dtype=int)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(epochs_range, train_acc, label='train')
    ax1.plot(epochs_range, test_acc, label='test')
    ax1.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.set_title('Accuracy')

    ax2.plot(epochs_range, train_loss, label='train')
    ax2.plot(epochs_range, test_loss, label='test')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('Loss')

    plt.suptitle(f'{model_name} @ {dataset_name}')

    plt.savefig(plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, default='SmallMLP', help=f'architecture to use',
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset to use',
                        choices=list(DATASETS.keys()))

    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='whether to use GPU (cuda) for accelerated computations or not')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='specify to use multiprocessing for accelerating computations on CPU '
                             '(note, it is impossible to use multiprocessing with GPU)')
    parser.add_argument('--processes', type=int, default=4,
                        help='how many processes to use for multiprocessing')

    parser.add_argument('--epochs', type=int, default=10, help='train for EPOCHS epochs')
    parser.add_argument('--batch_size', type=int, default=60, help='batch size (for samples-level batching)')
    parser.add_argument('--particles', type=int, default=100, help='')
    parser.add_argument('--particles_batch_size', type=int, default=10, help='batch size '
                                                                             '(for particles-level batching)')

    parser.add_argument('--alpha', type=float, default=50, help='alpha from CBO dynamics')
    parser.add_argument('--sigma', type=float, default=0.4 ** 0.5, help='sigma from CBO dynamics')
    parser.add_argument('--l', type=float, default=1, help='lambda from CBO dynamics')
    parser.add_argument('--dt', type=float, default=0.1, help='dt from CBO dynamics')
    parser.add_argument('--anisotropic', type=bool, default=True, help='whether to use anisotropic or not')
    parser.add_argument('--eps', type=float, default=1e-5, help='threshold for additional random shift')
    parser.add_argument('--partial_update', type=bool, default=True, help='whether to use partial or full update')
    parser.add_argument('--cooling', type=bool, default=False, help='whether to apply cooling strategy')

    parser.add_argument('--build_plot', required=False, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--plot_path', required=False, type=str, default='demo.png',
                        help='path to save the resulting plot')

    parser.add_argument('--eval_freq', type=int, default=100, help='evaluate test accuracy every EVAL_FREQ '
                                                                   'samples-level batches')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')

    model = MODELS[args.model]()
    train_dataloader, test_dataloader = DATASETS[args.dataset](train_batch_size=args.batch_size,
                                                               test_batch_size=args.batch_size)
    device = args.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Cuda is unavailable. Using CPU instead.')
        device = 'cpu'
    use_multiprocessing = args.use_multiprocessing
    if device != 'cpu' and use_multiprocessing:
        print('Unable to use multiprocessing on GPU')
        use_multiprocessing = False
    device = torch.device(device)

    print(f'Training {args.model} @ {args.dataset}')
    start_time = time.time()
    result = train(model, train_dataloader, test_dataloader, device, use_multiprocessing, args.processes,
                   args.epochs, args.particles, args.particles_batch_size,
                   args.alpha, args.sigma, args.l, args.dt, args.anisotropic, args.eps, args.partial_update,
                   args.cooling,
                   args.eval_freq)
    print(f'Elapsed time: {time.time() - start_time} seconds')
    if args.build_plot:
        build_plot(args.epochs, args.model, args.dataset, args.plot_path,
                   *result)
