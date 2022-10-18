import torch.nn as nn

small_mlp_model = nn.Sequential(
    nn.Flatten(1, 3),
    nn.Linear(28 ** 2, 10),
    nn.BatchNorm1d(10, affine=False),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.BatchNorm1d(10, affine=False),
    nn.Linear(10, 10),
    nn.LogSoftmax(),
)
