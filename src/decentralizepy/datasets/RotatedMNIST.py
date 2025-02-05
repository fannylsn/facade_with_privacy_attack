import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from decentralizepy.datasets.RotatedDataset import RotatedDataset
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

NUM_CLASSES = 10


class RotatedMNIST(RotatedDataset):
    """
    Class for the Rotated MNIST dataset
    """

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        random_seed: int = 1234,
        only_local=False,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size="",
        validation_source="",
        validation_size="",
        number_of_clusters: int = 1,
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        random_seed : int, optional
            Random seed for the dataset
        only_local : bool, optional
            True if the dataset needs to be partioned only among local procs, False otherwise
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64
        validation_source : str, optional
            Source of the validation set. Can be one of 'train' or 'test'
        validation_size : int, optional
            size of the test set used as validation set
        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            random_seed,
            only_local,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
            validation_source,
            validation_size,
            number_of_clusters,
        )

        self.num_classes = NUM_CLASSES

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                self.get_rotation_transform(),
            ]
        )

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def get_dataset_object(self, train: bool = True) -> torch.utils.data.Dataset:
        """Get the dataset object from torchvision or the filesystem."""
        dataset = torchvision.datasets.MNIST(
            root=self.train_dir, train=train, download=True, transform=self.transform
        )
        return dataset


class ConvNet(Model):
    """Model copied from the PyTorch tutorial on MNIST.
    Has 1'199'882 trainable parameters.
    Args:
        Model (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class FFNet(Model):
    """Same model as in the IFCA paper. suposed to achieve 95.05% accuracy on MNIST."""

    def __init__(self, h1=200):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # F.log_softmax(x, dim=1)


class FFNetSplit(Model):
    """Same model as in the IFCA paper. suposed to achieve 95.05% accuracy on MNIST."""

    HEAD_PARAM = ["fc2"]

    def __init__(self, h1=200):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # F.log_softmax(x, dim=1)

    def key_in_head(self, key):
        isin = False
        for ky in self.HEAD_PARAM:
            if ky in key:
                isin = True
        return isin


class LeNet(Model):
    """
    Class for a LeNet Model for CIFAR10
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 10 output classes

        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.gn1 = nn.GroupNorm(2, 32)
        self.conv2 = nn.Conv2d(32, 32, 5, padding="same")
        self.gn2 = nn.GroupNorm(2, 32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.gn3 = nn.GroupNorm(2, 64)
        self.fc1 = nn.Linear(64 * 4 * 4, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
