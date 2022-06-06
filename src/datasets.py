from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets

from src import DATASETS_DIR

"""
A list of helper functions for loading the popular datasets. By default, the datasets are downloaded and saved 
to the `saved_datasets` folder.
"""


def load_mnist_dataloaders(train_batch_size, test_batch_size, data_dir=None):
    """
    Downloads and loads the MNIST datasets. Returns the dataloaders for the train and test subsets respectively.
    :param train_batch_size: train dataloader batch size.
    :param test_batch_size: test dataloader batch size.
    :param data_dir: directory to save the downloaded dataset to. Optional.
    :return: train and test MNIST dataloaders (respectively).
    """
    if data_dir is None:
        data_dir = DATASETS_DIR
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        # Copied from this example https://github.com/pytorch/examples/blob/main/mnist/main.py#L114
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=mnist_transforms)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=mnist_transforms)
    train_dataloader = DataLoader(mnist_train, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=True)
    return train_dataloader, test_dataloader
