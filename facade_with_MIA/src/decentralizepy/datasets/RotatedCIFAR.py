import logging
from typing import List

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from decentralizepy.datasets.RotatedDataset import RotatedDataset
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

NUM_CLASSES = 10
CLASSES = {
    0: "plane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

class RotatedCIFAR(RotatedDataset):
    """
    Class for the Rotated CIFAR dataset
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
        top_k_acc=1,
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
            top_k_acc=top_k_acc,
        )
        self.partition_niid = partition_niid
        self.alpha = alpha
        self.shards = shards

        self.num_classes = NUM_CLASSES

        self.transform_dict = {
            cid: torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    self.get_rotation_transform(cid),
                ]
            )
            for cid in range(self.number_of_clusters)
        }

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def get_dataset_object(self, train: bool = True, cluster_id=None) -> torch.utils.data.Dataset:
        """Get the dataset object from torchvision or the filesystem."""
        cid = cluster_id if cluster_id is not None else self.cluster
        dataset = torchvision.datasets.CIFAR10(
            root=self.train_dir, train=train, download=True, transform=self.transform_dict[cid])
        return dataset


class LeNet(Model):
    """
    Class for a LeNet Model for CIFAR10 : 121'609 params
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791

    """
    # Uncomment the following lines to add/remove layers to/from the head 
    HEAD_LAYERS = ["fc1.weight", "fc1.bias"]
    # HEAD_LAYERS = []
    # HEAD_LAYERS = ["conv3.weight", "conv3.bias", "fc1.weight", "fc1.bias"]
    # HEAD_LAYERS = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias", "fc1.weight", "fc1.bias"]
    current_head = HEAD_LAYERS
    HEAD_BUFFERS = []
    current_head_buffers = HEAD_BUFFERS

    def __init__(self, feature_layer_name="conv3"):
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

    def deepcopy(self):
        """not deep carefull"""
        model_state = self.state_dict().copy()
        new_model = LeNet()
        new_model.load_state_dict(model_state)
        # new_model._register_hook()  # Make sure to re-register the hook
        return new_model

    def _register_hook(self):
        def hook(module, input, output):
            self.features = output

        for name, module in self.named_modules():
            if name == self.feature_layer_name:
                module.register_forward_hook(hook)

    def get_features(self):
        return self.features

    def reset_features(self):
        self.features = None

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

    def get_layers(self):
        """Get the blocks of the network."""
        return [self.conv1, self.conv2, self.conv3, self.fc1]

    def get_shared_layers(self):
        """Here define which layers are shared.

        Returns:
            List[torch.Tensor]: List of shared layer weights.
        """
        lays = []
        with torch.no_grad():
            for name, buffer in self.named_buffers():
                if name not in self.current_head_buffers:
                    lays.append(buffer.detach().clone())
            for name, param in self.named_parameters():
                if name not in self.current_head:
                    lays.append(param.detach().clone())
        return lays

    def set_shared_layers(self, shared_layers: List[nn.Module]):
        """Set the shared layers.

        Args:
            shared_layers (List[torch.Tensor]): List of shared layer weights.
        """
        shared_layers = shared_layers.copy()
        with torch.no_grad():
            for name, buffer in self.named_buffers():
                if name not in self.current_head_buffers:
                    buffer.copy_(shared_layers.pop(0))
            for name, param in self.named_parameters():
                if name not in self.current_head:
                    param.copy_(shared_layers.pop(0))
        assert (
            len(shared_layers) == 0
        ), "The shared_layers list should be empty after setting."

    def freeze_body(self):
        """Freeze the body of the network."""
        logging.debug("Freezing body of the network")
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False

    @classmethod
    def set_share_all(cls):
        # logging.info("Setting all layers to be shared")
        cls.current_head = []
        cls.current_head_buffers = []

    @classmethod
    def set_share_core(cls):
        # logging.info("Setting core layers to be shared")
        cls.current_head = cls.HEAD_LAYERS
        cls.current_head_buffers = cls.HEAD_BUFFERS


class LeNetSplit(Model):
    """
    Class for a LeNet Model for CIFAR10, used for DEPRL
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791

    """

    HEAD_PARAM = ["classifier"]

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
        self.classifier = nn.Sequential(nn.Linear(64 * 4 * 4, NUM_CLASSES))

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
        x = self.classifier(x)
        return x

    def key_in_head(self, key):
        isin = False
        for ky in self.HEAD_PARAM:
            if ky in key:
                isin = True
        return isin

    def deepcopy(self):
        """not deep carefull"""
        model_state = self.state_dict().copy()
        new_model = LeNetSplit()
        new_model.load_state_dict(model_state)
        return new_model


class LeNetCore(Model):
    """
    Class for a LeNet Model for CIFAR10
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791

    """

    def __init__(self, feature_layer_name="conv3"):
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
        return x


class LeNetSharedCore(nn.Module):
    def __init__(self, core_model, num_classes):
        super().__init__()
        self.core_model = core_model
        self.head = nn.Linear(64 * 4 * 4, NUM_CLASSES)

    def forward(self, x):
        x = self.core_model(x)
        x = self.head(x)
        return x
