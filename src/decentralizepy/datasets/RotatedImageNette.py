import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18

from decentralizepy.datasets.RotatedDataset import RotatedDataset
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

NUM_CLASSES = 10
IMG_SIZE = 128
CROP_SIZE = 64


class RotatedImageNette(RotatedDataset):
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
        # transorm from https://discuss.pytorch.org/t/how-do-i-resize-imagenet-image-to-224-x-224/66979
        # shortest side is 160px
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(CROP_SIZE),
                torchvision.transforms.CenterCrop(CROP_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                self.get_rotation_transform(),
                # self.get_color_transform(),
            ]
        )

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def get_dataset_object(self, train: bool = True) -> torch.utils.data.Dataset:
        """Get the dataset from the filesystem.
        Use download_imagenette.py to download the dataset.
        """

        dataset_dir = os.path.join(self.train_dir, "imagenette2-160")
        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "val")
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset not found at {dataset_dir}")

        # Normalization

        # Normal transformation
        # augment = True
        # if self.augment:
        #     train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        #     val_trans = [transforms.ToTensor(), norm]
        if train:
            data = datasets.ImageFolder(train_dir, transform=self.transform)
        else:
            data = datasets.ImageFolder(test_dir, transform=self.transform)

        # if self.cluster == 1:
        #     import matplotlib.pyplot as plt

        #     dl = iter(torch.utils.data.DataLoader(data, 1))
        #     x = next(dl)[0][0]
        #     x = next(dl)[0][0]
        #     print(x)
        #     plt.imshow(x.permute(1, 2, 0))
        #     plt.savefig("test.png")
        #     print("yes")

        return data


class LeNet(Model):
    """
    Class for a LeNet Model for CIFAR10
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
            247'561 parameters

        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.gn1 = nn.GroupNorm(2, 32)
        self.conv2 = nn.Conv2d(32, 32, 5, padding="same")
        self.gn2 = nn.GroupNorm(2, 32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.gn3 = nn.GroupNorm(2, 64)
        self.fc1 = nn.Linear(64 * 8 * 8, NUM_CLASSES)

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

    @classmethod
    def set_share_core(cls):
        """Never used in imgnette"""
        pass


class LeNetSplit(Model):
    """
    Class for a LeNet Model for CIFAR10
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
        self.classifier = nn.Sequential(nn.Linear(64 * 8 * 8, NUM_CLASSES))

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
        new_model = ResNet8()
        new_model.load_state_dict(model_state)
        return new_model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet8(Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNet8, self).__init__()
        block = BasicBlock
        num_blocks = [1, 1, 1]
        self.num_classes = num_classes
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(8192, num_classes)
        self.linear2 = nn.Linear(8192, num_classes)
        self.emb = nn.Embedding(num_classes, num_classes)
        self.emb.weight = nn.Parameter(torch.eye(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # b*128*64*64
        out = self.layer2(out)  # b*256*32*32
        out = self.layer3(out)  # b*512*16*16
        self.inner = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        self.flatten_feat = out  # b*2048
        out = self.linear1(out)
        return out

    def get_attentions(self):
        inner_copy = self.inner.detach().clone()  # b*512*8*8
        inner_copy.requires_grad = True
        out = F.avg_pool2d(inner_copy, 4)  # b*512*2*2
        out = out.view(out.size(0), -1)  # b*2048
        out = self.linear1(out)  # b*num_classes
        losses = out.sum(dim=0)  # num_classes
        cams = []
        # import ipdb;ipdb.set_trace()
        # assert losses.shape ==self.num_classes
        for n in range(self.num_classes):
            loss = losses[n]
            self.zero_grad()
            if n < self.num_classes - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            grads_val = inner_copy.grad
            weights = grads_val.mean(dim=(2, 3), keepdim=True)  # b*512*1*1
            cams.append(F.relu((weights.detach() * self.inner).sum(dim=1)))  # b*8*8
        atts = torch.stack(cams, dim=1)
        return atts


class ResNet18(Model):
    def __init__(self, num_classes: int = 200, pretrained: bool = False):
        super().__init__()
        pretrained = False
        if pretrained:
            self.resnet = resnet18(num_classes=num_classes, weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)


class AlexNet(nn.Module):
    """This could also be nice.
    https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/alexnet.py#L8"""

    def __init__(self, num_classes: int = 200):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
