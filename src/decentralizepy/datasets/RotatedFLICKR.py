import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets

from decentralizepy.datasets.RotatedDataset import RotatedDataset
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

NUM_CLASSES = 41
CROP_SIZE = 64


class RotatedFLICKR(RotatedDataset):
    """
    Class for the Rotated flickr mammals dataset
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

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(CROP_SIZE),
                torchvision.transforms.CenterCrop(CROP_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                self.get_rotation_transform(),
            ]
        )

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def get_dataset_object(self, train: bool = True) -> torch.utils.data.Dataset:
        """Get the dataset object from torchvision or the filesystem."""

        dataset_dir = os.path.join(self.train_dir, "geo_animal")
        train_dir = os.path.join(dataset_dir, "train2")
        test_dir = os.path.join(dataset_dir, "test2")
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

        # import matplotlib.pyplot as plt

        # dl = iter(torch.utils.data.DataLoader(data, 1))
        # x = next(dl)[0][0]
        # x = next(dl)[0][0]
        # plt.imshow(x.permute(1, 2, 0))
        # plt.savefig("test.png")
        # print("yes")
        return data


# class GoogleLeNet(Model):
#     """
#     Class for google lenet model : 6'624'904 params
#     """

#     def __init__(self):
#         super().__init__()
#         self.model = torch.hub.load("pytorch/vision:v0.10.0", "googlenet", weights=GoogLeNet_Weights.DEFAULT)
#         self.model.fc = torch.nn.Linear(1024, NUM_CLASSES)  # override the last layer

#     def forward(self, x):
#         return self.model(x)


# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(
#             in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False
#         )
#         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out


# class DephwiseBasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super().__init__()
#         self.conv1 = DepthwiseSeparableConv(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = DepthwiseSeparableConv(planes, planes, stride=1)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 DepthwiseSeparableConv(in_planes, self.expansion * planes, stride=stride),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet8_888(Model):
#     """888'837 params
#     9.8-10.3s for 10 batches of 8 images
#     """

#     def __init__(self, num_classes=41):
#         super().__init__()
#         block = DephwiseBasicBlock
#         num_blocks = [2, 2, 2]
#         self.num_classes = num_classes
#         self.in_planes = 48  # Reduced from 64

#         self.conv1 = DepthwiseSeparableConv(3, 48, stride=1)
#         self.bn1 = nn.BatchNorm2d(48)
#         self.layer1 = self._make_layer(block, 96, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 192, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 384, num_blocks[2], stride=2)

#         self.linear1 = nn.Linear(384, 192)  # Reduced size
#         self.linear2 = nn.Linear(192, num_classes)
#         self.dropout = nn.Dropout(p=0.5)
#         self.emb = nn.Embedding(num_classes, num_classes)
#         self.emb.weight = nn.Parameter(torch.eye(num_classes))

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)  # b*48*128*128
#         out = self.layer2(out)  # b*96*64*64
#         out = self.layer3(out)  # b*192*32*32
#         self.inner = out
#         out = F.adaptive_avg_pool2d(out, (1, 1))  # b*384*1*1
#         out = out.view(out.size(0), -1)  # b*384

#         self.flatten_feat = out  # b*384
#         out = self.dropout(F.relu(self.linear1(out)))
#         out = self.linear2(out)
#         return out


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


# class ResNet8Bigger(Model):
#     """1'264'890 params
#     4.7-5.2s for 10 batches of 8 images
#     NO!
#     """

#     HEAD_LAYERS = ["linear2.weight", "linear2.bias"]  # "linear1.weight", "linear1.bias",

#     def __init__(self, num_classes=NUM_CLASSES):
#         super().__init__()
#         block = BasicBlock
#         num_blocks = [1, 1, 1]
#         self.num_classes = num_classes
#         self.in_planes = 64  # reduce from 128

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # reduce from 128
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # reduce from 256
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # reduce from 512
#         self.linear1 = nn.Linear(256, 128)  # 2048, 512
#         self.linear2 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(0.5)
#         self.emb = nn.Embedding(num_classes, num_classes)
#         self.emb.weight = nn.Parameter(torch.eye(num_classes))

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)  # b*64*32*32
#         out = self.layer2(out)  # b*128*16*16
#         out = self.layer3(out)  # b*256*8*8
#         self.inner = out
#         out = F.adaptive_avg_pool2d(out, (1, 1))  # b*256*1*1 NEW
#         # out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)

#         self.flatten_feat = out  # b*256
#         out = self.dropout(F.relu(self.linear1(out)))
#         out = self.linear2(out)
#         return out

#     # def get_attentions(self):
#     #     inner_copy = self.inner.detach().clone()  # b*512*8*8
#     #     inner_copy.requires_grad = True
#     #     out = F.avg_pool2d(inner_copy, 4)  # b*512*2*2
#     #     out = out.view(out.size(0), -1)  # b*2048
#     #     out = self.linear1(out)  # b*num_classes
#     #     losses = out.sum(dim=0)  # num_classes
#     #     cams = []
#     #     # import ipdb;ipdb.set_trace()
#     #     # assert losses.shape ==self.num_classes
#     #     for n in range(self.num_classes):
#     #         loss = losses[n]
#     #         self.zero_grad()
#     #         if n < self.num_classes - 1:
#     #             loss.backward(retain_graph=True)
#     #         else:
#     #             loss.backward()
#     #         grads_val = inner_copy.grad
#     #         weights = grads_val.mean(dim=(2, 3), keepdim=True)  # b*512*1*1
#     #         cams.append(F.relu((weights.detach() * self.inner).sum(dim=1)))  # b*8*8
#     #     atts = torch.stack(cams, dim=1)
#     #     return atts

#     def get_shared_layers(self) -> List[torch.Tensor]:
#         """Here define which layers are shared.

#         Returns:
#             List[torch.Tensor]: List of shared layer weights.
#         """
#         with torch.no_grad():
#             lays = []
#             for name, param in self.named_parameters():
#                 if param.requires_grad and name not in self.HEAD_LAYERS:
#                     lays.append(param.data)
#             return lays

#     def set_shared_layers(self, shared_layers: List[torch.Tensor]):
#         """Set the shared layers.

#         Args:
#             shared_layers (List[torch.Tensor]): List of shared layer weights.
#         """
#         shared_layers = shared_layers.copy()
#         for name, param in self.named_parameters():
#             if param.requires_grad and name not in self.HEAD_LAYERS:
#                 param.data = shared_layers.pop(0)
#         assert len(shared_layers) == 0, "Should be empty"

#     def deepcopy(self):
#         """not deep carefull"""
#         model_state = self.state_dict().copy()
#         new_model = ResNet8Bigger()
#         new_model.load_state_dict(model_state)
#         return new_model


class ResNet8(Model):
    """312'649 params with 1,1,1 blocks
    700'617 params with 2,2,2 blocks"""

    HEAD_LAYERS = [
        "layer2.0.conv1.weight",
        "layer2.0.bn1.weight",
        "layer2.0.bn1.bias",
        "layer2.0.conv2.weight",
        "layer2.0.bn2.weight",
        "layer2.0.bn2.bias",
        "layer2.0.shortcut.0.weight",
        "layer2.0.shortcut.1.weight",
        "layer2.0.shortcut.1.bias",  ### end layer2
        "layer3.0.conv1.weight",
        "layer3.0.bn1.weight",
        "layer3.0.bn1.bias",
        "layer3.0.conv2.weight",
        "layer3.0.bn2.weight",
        "layer3.0.bn2.bias",
        "layer3.0.shortcut.0.weight",
        "layer3.0.shortcut.1.weight",
        "layer3.0.shortcut.1.bias",  ### end layer3
        "linear1.weight",
        "linear1.bias",
    ]
    current_head = HEAD_LAYERS

    HEAD_BUFFERS = [
        "layer2.0.bn1.running_mean",
        "layer2.0.bn1.running_var",
        "layer2.0.bn1.num_batches_tracked",
        "layer2.0.bn2.running_mean",
        "layer2.0.bn2.running_var",
        "layer2.0.bn2.num_batches_tracked",
        "layer2.0.shortcut.1.running_mean",
        "layer2.0.shortcut.1.running_var",
        "layer2.0.shortcut.1.num_batches_tracked",
        "layer3.0.bn1.running_mean",
        "layer3.0.bn1.running_var",
        "layer3.0.bn1.num_batches_tracked",
        "layer3.0.bn2.running_mean",
        "layer3.0.bn2.running_var",
        "layer3.0.bn2.num_batches_tracked",
        "layer3.0.shortcut.1.running_mean",
        "layer3.0.shortcut.1.running_var",
        "layer3.0.shortcut.1.num_batches_tracked",
    ]
    current_head_buffers = HEAD_BUFFERS

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        block = BasicBlock
        num_blocks = [1, 1, 1]
        self.num_classes = num_classes
        self.in_planes = 32  # reduce from 128

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # reduce from 128
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)  # reduce from 256
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)  # reduce from 512
        self.linear1 = nn.Linear(128, num_classes)
        # self.dropout = nn.Dropout(0.5)
        # self.emb = nn.Embedding(num_classes, num_classes)
        # self.emb.weight = nn.Parameter(torch.eye(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # b*32*64*64
        out = self.layer2(out)  # b*64*32*32
        out = self.layer3(out)  # b*128*16*16
        # self.inner = out
        out = F.adaptive_avg_pool2d(out, (1, 1))  # b*256*1*1 NEW
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # b*128

        # self.flatten_feat = out
        # out = self.dropout(F.relu(self.linear1(out)))
        out = self.linear1(out)
        return out

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

    def get_shared_layers(self) -> List[torch.Tensor]:
        """Define which layers are shared.

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

    def set_shared_layers(self, shared_layers: List[torch.Tensor]):
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

    def deepcopy(self):
        """not deep carefull"""
        model_state = self.state_dict().copy()
        new_model = ResNet8()
        new_model.load_state_dict(model_state)
        return new_model


class ResNet8Split(Model):
    """312'649 params with 1,1,1 blocks
    700'617 params with 2,2,2 blocks"""

    HEAD_PARAM = [
        "layer2.0.conv1.weight",  ### begin params of second basic block
        "layer2.0.bn1.weight",
        "layer2.0.bn1.bias",
        "layer2.0.conv2.weight",
        "layer2.0.bn2.weight",
        "layer2.0.bn2.bias",
        "layer2.0.shortcut.0.weight",
        "layer2.0.shortcut.1.weight",
        "layer2.0.shortcut.1.bias",  ### end layer2
        "layer3.0.conv1.weight",
        "layer3.0.bn1.weight",
        "layer3.0.bn1.bias",
        "layer3.0.conv2.weight",
        "layer3.0.bn2.weight",
        "layer3.0.bn2.bias",
        "layer3.0.shortcut.0.weight",
        "layer3.0.shortcut.1.weight",
        "layer3.0.shortcut.1.bias",  ### end layer3
        "linear1.weight",
        "linear1.bias",  ### end ff layer
        "layer2.0.bn1.running_mean",  # down here not params but buffers
        "layer2.0.bn1.running_var",
        "layer2.0.bn1.num_batches_tracked",
        "layer2.0.bn2.running_mean",
        "layer2.0.bn2.running_var",
        "layer2.0.bn2.num_batches_tracked",
        "layer2.0.shortcut.1.running_mean",
        "layer2.0.shortcut.1.running_var",
        "layer2.0.shortcut.1.num_batches_tracked",
        "layer3.0.bn1.running_mean",
        "layer3.0.bn1.running_var",
        "layer3.0.bn1.num_batches_tracked",
        "layer3.0.bn2.running_mean",
        "layer3.0.bn2.running_var",
        "layer3.0.bn2.num_batches_tracked",
        "layer3.0.shortcut.1.running_mean",
        "layer3.0.shortcut.1.running_var",
        "layer3.0.shortcut.1.num_batches_tracked",
    ]

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        block = BasicBlock
        num_blocks = [1, 1, 1]
        self.num_classes = num_classes
        self.in_planes = 32  # reduce from 128

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # reduce from 128
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)  # reduce from 256
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)  # reduce from 512
        self.linear1 = nn.Linear(128, num_classes)
        # self.dropout = nn.Dropout(0.5)
        # self.emb = nn.Embedding(num_classes, num_classes)
        # self.emb.weight = nn.Parameter(torch.eye(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # b*32*64*64
        out = self.layer2(out)  # b*64*32*32
        out = self.layer3(out)  # b*128*16*16
        # self.inner = out
        out = F.adaptive_avg_pool2d(out, (1, 1))  # b*256*1*1 NEW
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # b*128

        # self.flatten_feat = out
        # out = self.dropout(F.relu(self.linear1(out)))
        out = self.linear1(out)
        return out

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
