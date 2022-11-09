import torch.nn as nn


class SmallMLP(nn.Module):
    def __init__(self):
        super(SmallMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(28 ** 2, 10),
            nn.BatchNorm1d(10, affine=False),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10, affine=False),
            nn.Linear(10, 10),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.model(x)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(120, 10))

    def forward(self, x):
        output = self.cnn(x)
        return self.fc(output.view(x.shape[0], -1))
