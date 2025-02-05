import logging
from typing import List
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import torch
from PIL import ImageEnhance
from decentralizepy.datasets.DatasetClustered import DatasetClustered
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

from torch import nn
from torch.nn import functional as F

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

class ColorShiftCIFAR(DatasetClustered):
    """
    Class for Color Transformed CIFAR dataset to simulate non-IID data among clients.
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

        self.num_classes = NUM_CLASSES

        self.color_assignments = self.get_color_assignments()

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def get_color_assignments(self):
        """
        Assigns color space transformations to clusters.

        Returns
        -------
        List[str]
            A list where each entry corresponds to a cluster's color transformation.
            Example: ['none', 'grayscale', 'sepia', 'high_saturation']
        """
        assignments = []
        for cluster_id in range(self.number_of_clusters):
            if cluster_id == 0:  # No transformation for the first cluster
                assignments.append("none")
            elif cluster_id == 1:  # Grayscale
                assignments.append("high_saturation")
            elif cluster_id == 2:  # Sepia
                assignments.append("sepia")
            elif cluster_id == 3:  # High saturation
                assignments.append("grayscale")
            else:  # Cycle through transformations for additional clusters
                transformation_types = ["none", "sepia", "high_saturation", "grayscale"]
                assignments.append(transformation_types[cluster_id % len(transformation_types)])
        return assignments

    def get_color_transform(self):
        """
        Returns the color transformation based on the cluster assignment.

        Returns
        -------
        torchvision.transforms.Lambda
        """
        color_config = self.color_assignments[self.cluster]

        if color_config == "none":
            return Lambda(lambda x: x)  # Identity transform
        elif color_config == "grayscale":
            return Lambda(lambda x: x.mean(dim=0, keepdim=True).expand_as(x))  # Convert to grayscale
        elif color_config == "sepia":
            return Lambda(self._apply_sepia)
        elif color_config == "high_saturation":
            return Lambda(self._increase_saturation)
        else:
            raise ValueError(f"Unsupported color transformation: {color_config}")

    def _apply_sepia(self, tensor):
        """
        Applies sepia tone to an image tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The tensor with sepia applied.
        """
        r, g, b = tensor[0], tensor[1], tensor[2]
        tr = 0.393 * r + 0.769 * g + 0.189 * b
        tg = 0.349 * r + 0.686 * g + 0.168 * b
        tb = 0.272 * r + 0.534 * g + 0.131 * b
        return torch.stack([tr, tg, tb]).clamp(0, 1)

    def _increase_saturation(self, tensor):
        """
        Increases saturation of an image tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The tensor with increased saturation.
        """
        tensor = tensor.permute(1, 2, 0).numpy()  # Convert to HWC
        image = torchvision.transforms.ToPILImage()(tensor)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(2.0)  # Double the saturation
        return torchvision.transforms.ToTensor()(image)

    def get_transform(self):
        """
        Combines standard CIFAR-10 transforms with the color transformation.

        Returns
        -------
        torchvision.transforms.Compose
        """
        base_transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return Compose([base_transforms, self.get_color_transform()])

    def get_dataset_object(self, train=True):
        """
        Returns the CIFAR dataset object.

        Parameters
        ----------
        train : bool
            True for training set, False for test set.

        Returns
        -------
        torchvision.datasets.CIFAR10
        """
        dataset = torchvision.datasets.CIFAR10(
            root=self.train_dir if train else self.test_dir,
            train=train,
            download=True,
            transform=self.get_transform(),
        )
        return dataset

    def load_trainset(self):
        """
        Loads the training set and partitions it.
        """
        logging.info("Loading training set.")
        trainset = self.get_dataset_object(train=True)

        # Extract validation set if needed
        if self.__validating__ and self.validation_source == "Train":
            logging.info("Extracting the validation set from the train set.")
            self.validationset, trainset = torch.utils.data.random_split(
                trainset,
                [self.validation_size, 1.0 - self.validation_size],
                generator=torch.Generator().manual_seed(self.random_seed),
            )
            logging.info(f"The validation set has {len(self.validationset)} samples.")

        # Partition the dataset
        cluster_len = len(trainset)
        if self.sizes is None:
            self.sizes = [[1 / self.number_of_clusters] * self.number_of_clusters]

        cluster_sizes = [sum(size) for size in self.sizes]
        self.cluster_data_partitioner = DataPartitioner(
            trainset, sizes=cluster_sizes, seed=self.random_seed
        )
        cluster_data = self.cluster_data_partitioner.use(self.cluster)

        data_sizes = [size / sum(self.sizes[self.cluster]) for size in self.sizes[self.cluster]]
        self.data_partitioner = DataPartitioner(
            cluster_data, sizes=data_sizes, seed=self.random_seed
        )
        self.trainset = self.data_partitioner.use(self.dataset_id)

        logging.info(f"The training set has {len(self.trainset)} samples.")

    def load_testset(self):
        """
        Loads the testing set.
        """
        logging.info("Loading testing set.")
        self.testset = self.get_dataset_object(train=False)

        # Extract validation set if needed
        if self.__validating__ and self.validation_source == "Test":
            logging.info("Extracting the validation set from the test set.")
            self.validationset, self.testset = torch.utils.data.random_split(
                self.testset,
                [self.validation_size, 1.0 - self.validation_size],
                generator=torch.Generator().manual_seed(self.random_seed),
            )
            logging.info(f"The validation set has {len(self.validationset)} samples.")
        
        logging.info(f"The test set has {len(self.testset)} samples.")




class LeNet(Model):
    """
    Class for a LeNet Model for CIFAR10 : 121'609 params
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791

    """

    HEAD_LAYERS = ["fc1.weight", "fc1.bias"]
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

        # Register hook for feature extraction
        # self.feature_layer_name = feature_layer_name
        # self._register_hook()
        # self.reset_features()

    # def deepcopy(self):
    #     with torch.no_grad():
    #         model = copy.deepcopy(self)
    #         model._register_hook()
    #         return model

    def deepcopy(self):
        """not deep carefull"""
        model_state = self.state_dict().copy()
        new_model = LeNet()
        new_model.load_state_dict(model_state)
        # new_model._register_hook()  # Make sure to re-register the hook
        return new_model

    # def shallowcopy(self):
    #     model = self.clone()
    #     model._register_hook()
    #     return model

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
        assert len(shared_layers) == 0, "The shared_layers list should be empty after setting."

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
