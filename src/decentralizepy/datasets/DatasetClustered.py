import logging
from random import Random
from typing import List, Union

import torch
from torch.utils.data import DataLoader

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model
from decentralizepy.utils_decpy.losses import FeatureAlignmentLoss


class DatasetClustered(Dataset):
    """
    This class defines the Dataset API for niid clustered data.
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
        )
        self.num_nodes = self.num_partitions  # more explicit
        logging.debug("Number of nodes: {}".format(self.num_nodes))
        logging.debug("Rank: {}".format(rank))
        logging.debug("Uid: {}".format(self.uid))
        self.num_classes = None  # to be set by the child class

        self.number_of_clusters = number_of_clusters
        self.assign_cluster()
        logging.info("Clusters idx: {}".format(self.clusters_idx))
        logging.info("Cluster: {}".format(self.cluster))

        # carefull, we dont duplicated one dataset per rotated clusted like IFCA.
        self.dataset_id = sum(
            [idx == self.cluster for idx in self.clusters_idx[: self.uid]]
        )  # id of the dataset in the cluster of the node

        self.top_k_acc = top_k_acc
        logging.debug(f"topk accuracy: {self.top_k_acc}")

    def assign_cluster(self):
        """Generate the cluster assignment for the current process."""
        rng = Random()
        rng.seed(self.random_seed)
        if self.sizes is None:
            self.clusters_idx = [
                i % self.number_of_clusters for i in range(self.num_nodes)
            ]
        else:
            self.clusters_idx = []
            for idx, size in enumerate(self.sizes):
                self.clusters_idx += [idx] * len(size)
        rng.shuffle(self.clusters_idx)
        self.cluster = self.clusters_idx[self.uid]

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        raise NotImplementedError

    def load_testset(self):
        """
        Loads the testing set.

        """
        raise NotImplementedError

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
            # torch.manual_seed(self.random_seed * self.uid)
            # x = next(iter(DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)))
            # logging.info(f"sample of call dataloader: {x[0][0,0,0]}")
            # torch.manual_seed(self.random_seed * self.uid)
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

    def get_validationset(self, batchsize=None, Shuffle=None) -> DataLoader:
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
        if batchsize is None:
            batchsize = self.test_batch_size
        if self.__validating__:
            return DataLoader(self.validationset, batch_size=batchsize, shuffle=Shuffle)
        raise RuntimeError("Validation set not initialized!")

        # def get_validationset(self, batch_size=None, shuffle=False) -> DataLoader:
        #     if batch_size is None:
        #         batch_size = self.test_batch_size
        #     if self.__validating__:
        #         return DataLoader(self.validationset, batch_size=batch_size, shuffle=shuffle)
        #     raise RuntimeError("Validation set not initialized!")

    def test(self, models: Union[List[Model], Model], loss_func):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        models : List[decentralizepy.models.Model]
            Models to be chosen from and evaluate on the best one
        loss_func : torch.nn.loss
            Loss function to use

        Returns
        -------
        tuple(float, float, int)

        """
        logging.debug("Evaluate model on the test set")
        loss_vals = []
        correct_preds_per_cls = []
        totals_pred_per_cls = []
        totals_correct = []
        totals_predicted = []

        only_one_model = False
        if not isinstance(models, list):
            only_one_model = True
            models = [models]

        for i, model in enumerate(models):
            model.eval()
            logging.debug("Model {} in evaluation mode.".format(i))
            # with fairness, one func per model
            if isinstance(loss_func, list):
                loss_func_ = loss_func[i]
            else:
                loss_func_ = loss_func

            correct_pred_per_cls = [0 for _ in range(self.num_classes)]
            total_pred_per_cls = [0 for _ in range(self.num_classes)]

            total_correct = 0
            total_predicted = 0

            with torch.no_grad():
                loss_val = 0.0
                count = 0
                for elems, labels in self.get_testset():
                    outputs = model(elems)
                    if isinstance(loss_func_, FeatureAlignmentLoss):
                        # forward pass for the other model to compute features
                        _ = loss_func_.model_other(elems)
                    loss_val += loss_func_(outputs, labels).item()
                    count += 1
                    _, predictions = torch.topk(outputs, self.top_k_acc)
                    # _, predictions = torch.max(outputs, 1)
                    for label, prediction in zip(labels, predictions):
                        # logging.debug("{} predicted as {}".format(label, prediction))
                        if label in prediction:  # top-k done here
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

        if only_one_model:
            # compatibility with the previous version
            return accuracy, final_loss_val

        return accuracy, final_loss_val, best_model_idx

    def validate(self, models: Union[List[Model], Model], loss_func):
        """
        Function to evaluate model on the validation dataset.

        Parameters
        ----------
        models : List[decentralizepy.models.Model]
            Models to be chosen from and evaluate on the best one
        loss : torch.nn.loss
            Loss function to use

        Returns
        -------
        tuple(float, float, int)

        """
        logging.debug("Evaluate model on the val set")
        loss_vals = []
        correct_preds_per_cls = []
        totals_pred_per_cls = []
        totals_correct = []
        totals_predicted = []

        only_one_model = False
        if not isinstance(models, list):
            only_one_model = True
            models = [models]

        for i, model in enumerate(models):
            model.eval()
            logging.debug("Model {} in evaluation mode.".format(i))

            # with fairness, one func per model
            if isinstance(loss_func, list):
                loss_func_ = loss_func[i]
            else:
                loss_func_ = loss_func

            correct_pred_per_cls = [0 for _ in range(self.num_classes)]
            total_pred_per_cls = [0 for _ in range(self.num_classes)]

            total_correct = 0
            total_predicted = 0

            with torch.no_grad():
                loss_val = 0.0
                count = 0
                for elems, labels in self.get_validationset():
                    outputs = model(elems)
                    if isinstance(loss_func_, FeatureAlignmentLoss):
                        # forward pass for the other model to compute features
                        _ = loss_func_.model_other(elems)
                    loss_val += loss_func_(outputs, labels).item()
                    count += 1
                    _, predictions = torch.topk(outputs, self.top_k_acc)
                    # _, predictions = torch.max(outputs, 1)
                    for label, prediction in zip(labels, predictions):
                        # logging.debug("{} predicted as {}".format(label, prediction))
                        if label in prediction:  # top-k done here
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
        loss_vals = [loss_val / count for loss_val in loss_vals]
        final_loss_val = loss_vals[best_model_idx]
        logging.info("Overall test accuracy is: {:.1f} %".format(accuracy))

        if only_one_model:
            # compatibility with the previous version
            return accuracy, final_loss_val

        return accuracy, final_loss_val, loss_vals, best_model_idx

    def compute_per_sample_loss(
        self,
        model: Model,
        loss_func,
        validation: bool = False,
        log_loss: bool = False,
        log_pred_true: bool = False,
    ):
        """
        Compute the per sample loss for the current model (the one that will be shared).

        Args:
            model (decentralizepy.models.Model): The model to evaluate.
            loss_func (torch.nn.loss): The loss function to use
            validation (bool): True if the validation set should be used, False otherwise
        """
        model.eval()
        if validation:
            dataset = self.get_validationset()
        else:
            dataset = self.get_testset()

        with torch.no_grad():
            per_sample_loss = []
            per_sample_pred = []
            per_sample_true = []
            if not log_loss and not log_pred_true:
                return per_sample_loss, per_sample_pred, per_sample_true
            for elems, labels in dataset:
                outputs = model(elems)
                if isinstance(loss_func, FeatureAlignmentLoss):
                    # forward pass for the other model to compute features
                    _ = loss_func.model_other(elems)
                loss_val = loss_func(outputs, labels)
                # _, predictions = torch.max(outputs, 1)
                _, predictions = torch.topk(outputs, self.top_k_acc)
                if log_loss:
                    per_sample_loss.extend(loss_val.tolist())
                if log_pred_true:
                    if self.top_k_acc > 1:
                        per_sample_pred.append(
                            predictions.tolist()
                        )  # pred is list of list
                    else:
                        per_sample_pred.extend(predictions.tolist())
                    per_sample_true.extend(labels.tolist())

        return per_sample_loss, per_sample_pred, per_sample_true
