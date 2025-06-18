import logging

import torch
import torchvision

from decentralizepy.datasets.DatasetClustered import DatasetClustered
from decentralizepy.datasets.Partitioner import DataPartitioner
from torch.utils.data import DataLoader


class RotatedDataset(DatasetClustered):
    """
    Class for Rotated dataset (used to simulate non IID data).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation_angle = 0  # Initialize rotation angle attribute

    def get_rotation_transform(self, cluster_id=None) -> torchvision.transforms.RandomRotation:
        """
        Returns a rotation transform based on the cluster assignment

        Returns
        -------
        torchvision.transforms.RandomRotation

        """
        if self.number_of_clusters == 1:
            self.rotation_angle = 0
        elif self.number_of_clusters == 2:
            self.rotation_angle = 180 
        elif self.number_of_clusters == 3 or self.number_of_clusters == 4:
            self.rotation_angle = 90 
        else:
            raise ValueError(
                "Rotation transform not implemented for {} clusters".format(self.number_of_clusters)
                )

        cid = cluster_id if cluster_id is not None else self.cluster
        rotation = self.rotation_angle * cid
        logging.info(f"Cluster {cid} applying rotation: {rotation}")

        return torchvision.transforms.RandomRotation(degrees=[rotation, rotation])

    def get_color_transform(self) -> torchvision.transforms.ColorJitter:
        if self.number_of_clusters == 2:
            if self.cluster == 1:
                return torchvision.transforms.Grayscale(num_output_channels=3)
            else:
                return NullTransform()

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        trainset_dict = {
            cid : self.get_dataset_object(train=True, cluster_id=cid) for cid in range(self.number_of_clusters)
        }

        if self.__validating__ and self.validation_source == "Train":
            logging.info("Extracting the validation set from the train set.")
            updated_trainset_dict = {}

            for cid in range(self.number_of_clusters):
                validationset, trainset = torch.utils.data.random_split(
                    trainset_dict[cid],
                    [self.validation_size, 1 - self.validation_size],
                    torch.Generator().manual_seed(self.random_seed),
                )
                if cid == self.cluster:
                    self.validationset = validationset
                    logging.info(f"The validation set for cluster {cid} has {len(validationset)} samples.")
                updated_trainset_dict[cid] = trainset
            
            trainset_dict = updated_trainset_dict

        c_len = len(trainset_dict[self.cluster])

        if self.sizes is None:
            self.sizes = []
            for i in range(self.number_of_clusters):
                node_in_cluster_i = sum([idx == i for idx in self.clusters_idx])
                e = c_len // node_in_cluster_i
                frac = e / c_len
                self.sizes.append([frac] * node_in_cluster_i)
                self.sizes[i][-1] += 1.0 - frac * node_in_cluster_i
            logging.debug("Size fractions: {}".format(self.sizes))

        print(self.random_seed)
        cluster_sizes = [sum(sizes) for sizes in self.sizes]
        cluster_data_partitioner_dict = {
            cid: DataPartitioner(
                trainset_dict[cid],
                sizes=cluster_sizes,
                seed=self.random_seed,
            ) for cid in range(self.number_of_clusters)
        }
        cluster_data_dict = {
            cid: cluster_data_partitioner_dict[cid].use(cid) for cid in range(self.number_of_clusters)
        }

        data_sizes_dict = {
            cid: [sizes / cluster_sizes[cid] for sizes in self.sizes[cid]]
            for cid in range(self.number_of_clusters)
        }

        self.data_partitioner_dict = {
            cid: DataPartitioner(cluster_data_dict[cid], sizes=data_sizes_dict[cid], seed=self.random_seed) 
            for cid in range(self.number_of_clusters)
        }

        self.trainset = self.data_partitioner_dict[self.cluster].use(self.dataset_id)
        logging.info(f"The training set has {len(self.trainset)} samples.")


    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        self.testset_dict = {
            cid : self.get_dataset_object(train=False, cluster_id=cid) for cid in range(self.number_of_clusters)
        }
        
        if self.__validating__ and self.validation_source == "Test":
            logging.info("Extracting the validation set from the test set.")
            updated_testset_dict = {}

            for cid in range(self.number_of_clusters):
                validationset, testset = torch.utils.data.random_split(
                    self.testset_dict[cid],
                    [self.validation_size, 1 - self.validation_size],
                    torch.Generator().manual_seed(self.random_seed),
                )
                if cid == self.cluster:
                    self.validationset = validationset
                    logging.info(f"The validation set for cluster {cid} has {len(validationset)} samples.")
                updated_testset_dict[cid] = testset

            self.testset_dict = updated_testset_dict

        logging.info(f"The test set has {len(self.testset_dict[self.cluster])} samples.")


    def get_trainset(self, batch_size=1, shuffle=False, node_id=None):
        """
        Function to get the training set
        """
        if self.__training__:
            # to get the dataset of another node when performing a training attack
            if node_id is not None:
                cluster_id = self.clusters_idx[node_id]
                dataset_id = sum([idx == cluster_id for idx in self.clusters_idx[:node_id]])
                return DataLoader(self.data_partitioner_dict[cluster_id].use(dataset_id), batch_size, shuffle)
            else:
                return super().get_trainset(batch_size, shuffle)
            
        raise RuntimeError("Training set not initialized!")
    
    
    def get_testset(self, cluster_id=None, attack=False) -> DataLoader:
        """
        Function to get the test set
        """
        if self.__testing__ or attack:
            cid = cluster_id if cluster_id is not None else self.cluster
            return DataLoader(self.testset_dict[cid], batch_size=self.test_batch_size)
        raise RuntimeError("Test set not initialized!")


class NullTransform:
    def __call__(self, sample):
        return sample
