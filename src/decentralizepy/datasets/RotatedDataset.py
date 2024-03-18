import logging
from random import Random
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model


class RotatedDataset(Dataset):
    """
    Class for Rotated dataset
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
        self.dataset_id = sum(
            [idx == self.cluster for idx in self.clusters_idx[: self.rank]]
        )  # id of the dataset in the cluster of the node

    def assign_cluster(self):
        """Generate the cluster assignment for the current process."""
        rng = Random()
        rng.seed(self.random_seed)
        if self.sizes is None:
            self.clusters_idx = [
                i % self.number_of_clusters for i in range(self.num_partitions)
            ]
        else:
            self.clusters_idx = []
            for idx, size in enumerate(self.sizes):
                self.clusters_idx += [idx] * len(size)
        rng.shuffle(self.clusters_idx)
        self.cluster = self.clusters_idx[self.rank]

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
            return torchvision.transforms.RandomRotation(
                degrees=[180 * self.cluster, 180 * self.cluster]
            )
        elif self.number_of_clusters == 4:
            return torchvision.transforms.RandomRotation(
                degrees=[90 * self.cluster, 90 * self.cluster]
            )
        else:
            raise ValueError(
                "Rotation transform not implemented for {} clusters".format(
                    self.number_of_clusters
                )
            )

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
                e = c_len // node_in_cluster_i
                frac = e / c_len
                self.sizes.append([frac] * node_in_cluster_i)
                self.sizes[i][-1] += 1.0 - frac * node_in_cluster_i
            logging.debug("Size fractions: {}".format(self.sizes))

        self.data_partitioner = DataPartitioner(
            trainset,
            sizes=self.sizes[self.cluster],
            seed=self.random_seed,
        )
        self.trainset = self.data_partitioner.use(self.dataset_id)

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

    def get_dataset_object(self, train: bool = True) -> torch.utils.data.Dataset:
        """Get the dataset object from torchvision or the filesystem."""
        raise NotImplementedError

    def get_trainset(self, batch_size=1, shuffle=False) -> DataLoader:
        """
        Function to get the training set

        Parameters
        ----------
        batch_size : int, optional
            Batch size for learning

        Returns
        -------
        torch.utils.data.Dataloader

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        if self.__training__:
            return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        raise RuntimeError("Training set not initialized!")

    def get_testset(self) -> DataLoader:
        """
        Function to get the test set

        Returns
        -------
        torch.utils.data.Dataloader

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__testing__:
            return DataLoader(self.testset, batch_size=self.test_batch_size)
        raise RuntimeError("Test set not initialized!")

    def get_validationset(self) -> DataLoader:
        """
        Function to get the validation set

        Returns
        -------
        torch.utils.data.Dataloader

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__validating__:
            return DataLoader(self.validationset, batch_size=self.test_batch_size)
        raise RuntimeError("Validation set not initialized!")

    def test(self, models: List[Model], loss_func):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to use

        Returns
        -------
        tuple(float, float)

        """
        logging.debug("Evaluate model on the test set")
        loss_vals = []
        correct_preds_per_cls = []
        totals_pred_per_cls = []
        totals_correct = []
        totals_predicted = []

        for i, model in enumerate(models):
            model.eval()
            logging.debug("Model {} in evaluation mode.".format(i))

            correct_pred_per_cls = [0 for _ in range(self.num_classes)]
            total_pred_per_cls = [0 for _ in range(self.num_classes)]

            total_correct = 0
            total_predicted = 0

            with torch.no_grad():
                loss_val = 0.0
                count = 0
                for elems, labels in self.get_testset():
                    outputs = model(elems)
                    loss_val += loss_func(outputs, labels).item()
                    count += 1
                    _, predictions = torch.max(outputs, 1)
                    for label, prediction in zip(labels, predictions):
                        logging.debug("{} predicted as {}".format(label, prediction))
                        if label == prediction:
                            correct_pred_per_cls[label] += 1
                            total_correct += 1
                        total_pred_per_cls[label] += 1
                        total_predicted += 1

            loss_vals.append(loss_val)
            correct_preds_per_cls.append(correct_pred_per_cls)
            totals_pred_per_cls.append(total_pred_per_cls)
            totals_correct.append(total_correct)
            totals_predicted.append(total_predicted)

        best_model_idx = loss_vals.index(min(loss_vals))

        logging.debug(f"Predicted on the test set. Best model is {best_model_idx}")

        for key, value in enumerate(correct_preds_per_cls[best_model_idx]):
            if totals_pred_per_cls[best_model_idx][key] != 0:
                accuracy = 100 * float(value) / totals_pred_per_cls[best_model_idx][key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = (
            100
            * float(totals_correct[best_model_idx])
            / totals_predicted[best_model_idx]
        )
        final_loss_val = loss_vals[best_model_idx] / count
        logging.info("Overall test accuracy is: {:.1f} %".format(accuracy))
        return accuracy, final_loss_val, best_model_idx

    def validate(self, models: List[Model], loss_func):
        """
        Function to evaluate model on the validation dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to use

        Returns
        -------
        tuple(float, float)

        """
        logging.debug("Evaluate model on the test set")
        loss_vals = []
        correct_preds_per_cls = []
        totals_pred_per_cls = []
        totals_correct = []
        totals_predicted = []

        for i, model in enumerate(models):
            model.eval()
            logging.debug("Model {} in evaluation mode.".format(i))

            correct_pred_per_cls = [0 for _ in range(self.num_classes)]
            total_pred_per_cls = [0 for _ in range(self.num_classes)]

            total_correct = 0
            total_predicted = 0

            with torch.no_grad():
                loss_val = 0.0
                count = 0
                for elems, labels in self.get_validationset():
                    outputs = model(elems)
                    loss_val += loss_func(outputs, labels).item()
                    count += 1
                    _, predictions = torch.max(outputs, 1)
                    for label, prediction in zip(labels, predictions):
                        logging.debug("{} predicted as {}".format(label, prediction))
                        if label == prediction:
                            correct_pred_per_cls[label] += 1
                            total_correct += 1
                        total_pred_per_cls[label] += 1
                        total_predicted += 1

            loss_vals.append(loss_val)
            correct_preds_per_cls.append(correct_pred_per_cls)
            totals_pred_per_cls.append(total_pred_per_cls)
            totals_correct.append(total_correct)
            totals_predicted.append(total_predicted)

        best_model_idx = loss_vals.index(min(loss_vals))

        logging.debug(f"Predicted on the val set. Best model is {best_model_idx}")

        for key, value in enumerate(correct_preds_per_cls[best_model_idx]):
            if totals_pred_per_cls[best_model_idx][key] != 0:
                accuracy = 100 * float(value) / totals_pred_per_cls[best_model_idx][key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = (
            100
            * float(totals_correct[best_model_idx])
            / totals_predicted[best_model_idx]
        )
        final_loss_val = loss_vals[best_model_idx] / count
        logging.info("Overall test accuracy is: {:.1f} %".format(accuracy))
        return accuracy, final_loss_val, best_model_idx
