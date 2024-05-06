import logging

import torch
import torchvision

from decentralizepy.datasets.DatasetClustered import DatasetClustered
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping


class RotatedDataset(DatasetClustered):
    """
    Class for Rotated dataset (used to simulate non IID data).
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
        )

        self.num_classes = None  # to be set by the child class

        self.number_of_clusters = number_of_clusters
        self.assign_cluster()

        # to have the full dataset for each cluster (IFCA were doing it this way with rotations)
        self.duplicate_datasets = False
        if self.duplicate_datasets:
            self.dataset_id = sum(
                [idx == self.cluster for idx in self.clusters_idx[: self.rank]]
            )  # id of the dataset in the cluster of the node
        else:
            self.dataset_id = self.rank

    def get_rotation_transform(self) -> torchvision.transforms.RandomRotation:
        """
        Returns a rotation transform based on the cluster assignment

        Returns
        -------
        torchvision.transforms.RandomRotation

        """
        if self.number_of_clusters == 1:
            return torchvision.transforms.RandomRotation(degrees=[0, 0])
        elif self.number_of_clusters == 2:
            return torchvision.transforms.RandomRotation(degrees=[180 * self.cluster, 180 * self.cluster])
        elif self.number_of_clusters == 4:
            return torchvision.transforms.RandomRotation(degrees=[90 * self.cluster, 90 * self.cluster])
        else:
            raise ValueError("Rotation transform not implemented for {} clusters".format(self.number_of_clusters))

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        trainset = self.get_dataset_object(train=True)
        # try to overfit a small part (sanity check) !!
        # trainset = torch.utils.data.Subset(trainset, range(0, 160))

        # in case the val set is extracted from the train set
        if self.__validating__ and self.validation_source == "Train":
            logging.info("Extracting the validation set from the train set.")
            self.validationset, trainset = torch.utils.data.random_split(
                trainset,
                [self.validation_size, 1 - self.validation_size],
                torch.Generator().manual_seed(self.random_seed),
            )

        c_len = len(trainset)

        if self.sizes is None:
            self.sizes = []
            for i in range(self.number_of_clusters):
                node_in_cluster_i = sum([idx == i for idx in self.clusters_idx])
                if self.duplicate_datasets:
                    e = c_len // node_in_cluster_i
                    frac = e / c_len
                    self.sizes.append([frac] * node_in_cluster_i)
                    self.sizes[i][-1] += 1.0 - frac * node_in_cluster_i
                else:
                    e = c_len // self.num_nodes
                    frac = e / c_len
                    self.sizes.append([frac] * node_in_cluster_i)
                    self.sizes[i][-1] += (1.0 / self.number_of_clusters) - frac * node_in_cluster_i
            logging.debug("Size fractions: {}".format(self.sizes))

        if self.duplicate_datasets:
            self.data_partitioner = DataPartitioner(
                trainset,
                sizes=self.sizes[self.cluster],
                seed=self.random_seed,
            )
        else:
            # current way
            self.data_partitioner = DataPartitioner(
                trainset,
                sizes=[x for sublist in self.sizes for x in sublist],
                seed=self.random_seed,
            )

        self.trainset = self.data_partitioner.use(self.dataset_id)
        logging.info(f"The training set has {len(self.trainset)} samples.")

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        testset = self.get_dataset_object(train=False)
        # try to overfit a small part (sanity check) !!
        # testset = torch.utils.data.Subset(testset, range(0, 10))

        if self.__validating__ and self.validation_source == "Test":
            logging.info("Extracting the validation set from the test set.")
            self.validationset, self.testset = torch.utils.data.random_split(
                testset,
                [self.validation_size, 1 - self.validation_size],
                torch.Generator().manual_seed(self.random_seed),
            )
