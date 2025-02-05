import logging

import numpy as np
import torch

from decentralizepy.datasets.DatasetClustered import DatasetClustered
from decentralizepy.datasets.Partitioner import (
    DataPartitioner,
    DirichletDataPartitioner,
    KShardDataPartitioner,
    Partition,
)
from decentralizepy.mappings.Mapping import Mapping


class LabelShiftDataset(DatasetClustered):
    """
    This class defines the labelshifted datasets.
    All datasets must follow this API.

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
        partition_niid: str = "dirichlet",
        alpha: float = 0.1,
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

        self.partition_niid = partition_niid
        self.alpha = alpha

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
        self.orig_trainset_len = c_len

        if self.sizes is None:
            self.sizes = []
            for i in range(self.number_of_clusters):
                node_in_cluster_i = sum([idx == i for idx in self.clusters_idx])
                e = c_len // node_in_cluster_i
                frac = e / c_len
                self.sizes.append([frac] * node_in_cluster_i)
                self.sizes[i][-1] += 1.0 - frac * node_in_cluster_i
            logging.debug("Size fractions: {}".format(self.sizes))

        if self.partition_niid == "dirichlet":
            self.data_partitioner = DirichletDataPartitioner(
                trainset,
                sizes=self.sizes,  # give all the sizes (just need len(sizes))
                seed=self.random_seed,
                alpha=self.alpha,
                num_classes=self.num_classes,
            )
        elif self.partition_niid == "kshard":
            # order the data !!
            train_data = {key: [] for key in range(self.num_classes)}
            for x, y in trainset:
                train_data[y].append(x)

            # randomize class order
            np.random.seed(self.random_seed)
            random_class_order = np.random.permutation(self.num_classes)
            all_trainset = []
            for y in random_class_order:
                all_trainset.extend([(a, y) for a in train_data[y]])

            # kshard_sizes = [sum(fracs) / self.number_of_clusters for fracs in self.sizes]
            kshard_sizes = [1.0 / self.number_of_clusters] * self.number_of_clusters
            # CHECK sizes
            self.data_partitioner = KShardDataPartitioner(
                all_trainset, kshard_sizes, shards=1, seed=self.random_seed
            )

        elif self.partition_niid == "animals_and_vehicules":
            raise NotImplementedError
        else:
            raise ValueError("NIID type not supported")

        # get the dirichlet partition for the cluster, then repartition this data
        cluster_data = self.data_partitioner.use(self.cluster)
        logging.debug(f"cluster {self.cluster} has {len(cluster_data)} samples")
        self.cluster_distribution = self.compute_cluster_ditribution(cluster_data)
        logging.debug(
            f"adapting sizes for cluster {self.cluster} with sizes {self.sizes[self.cluster]}"
        )
        # adapted_sizes = [self.cluster * frac for frac in self.sizes[self.cluster]]
        # logging.debug(f"adapted sizes: {adapted_sizes}")
        self.data_sub_partitioner = DataPartitioner(
            cluster_data,
            sizes=self.sizes[self.cluster],
            seed=self.random_seed,
        )
        self.trainset = self.data_sub_partitioner.use(self.dataset_id)
        logging.info(
            f"Dataset ID {self.dataset_id} of cluser {self.cluster} has {len(self.trainset)} samples in trainset."
        )

        # from torch.utils.data import DataLoader
        # d = DataLoader(self.trainset, batch_size=100, shuffle=True)
        # x, y = next(iter(d))
        # import matplotlib.pyplot as plt

        # plt.hist(y.numpy(), bins=self.num_classes, range = (0,9))
        # plt.savefig("trainset_batch.png")
        # pass

    def compute_cluster_ditribution(self, cluster_data):
        """
        Compute the distribution of the cluster data

        Parameters
        ----------
        cluster_data : list
            The data of the cluster

        Returns
        -------
        dict
            The distribution of the cluster data

        """
        cluster_distribution = {key: 0 for key in range(self.num_classes)}
        for _, y in cluster_data:
            cluster_distribution[y] += 1
        cluster_distribution = {
            key: value / len(cluster_data)
            for key, value in cluster_distribution.items()
        }
        return cluster_distribution

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        testset = self.get_dataset_object(train=False)
        # not constant size for dirichlet -> need to scale according to train repartition
        new_idx = self.apply_distribution(
            testset,
            self.cluster_distribution,
            len(testset) * len(self.trainset) // self.orig_trainset_len,
        )
        testset = Partition(testset, new_idx)
        # try to overfit a small part (sanity check) !!
        # testset = torch.utils.data.Subset(testset, range(0, 10))

        if self.__validating__ and self.validation_source == "Test":
            logging.info("Extracting the validation set from the test set.")
            self.validationset, self.testset = torch.utils.data.random_split(
                testset,
                [self.validation_size, 1 - self.validation_size],
                torch.Generator().manual_seed(self.random_seed),
            )

    def apply_distribution(self, dataset, label_distribution, number_of_draws):
        """
        Apply the distribution to the dataset

        Parameters
        ----------
        dataset : torchvision.datasets
            The dataset
        label_distribution : dict
            The distribution of the labels
        number_of_draws : int
            The number of draws

        Returns
        -------
        list
            The dataset with the distribution applied

        """
        np.random.seed(self.random_seed)
        new_indexes = []
        exact_num = True
        if exact_num:
            for class_, prob in label_distribution.items():
                draws = int(prob * number_of_draws)
                if draws == 0:
                    continue
                idx_class = np.where(np.array(dataset.targets) == class_)[0]
                new_idx = np.random.choice(idx_class, draws, replace=False)
                new_indexes.extend(new_idx)

        else:
            pass
            # class_idx = torch.multinomial(torch.Tensor(label_distribution), number_of_draws, replacement=True)

        np.random.shuffle(new_indexes)
        return new_indexes
