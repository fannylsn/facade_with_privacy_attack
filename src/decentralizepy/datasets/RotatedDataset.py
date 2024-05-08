import logging

import torch
import torchvision

from decentralizepy.datasets.DatasetClustered import DatasetClustered
from decentralizepy.datasets.Partitioner import DataPartitioner


class RotatedDataset(DatasetClustered):
    """
    Class for Rotated dataset (used to simulate non IID data).
    """

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
