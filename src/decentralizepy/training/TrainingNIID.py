import logging

import torch

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.training.Training import Training


class TrainingNIID(Training):
    """
    This class just adds a few function for per sample computaions.
    """

    def train_full(self, dataset):
        """
        Redefine to remoove logs...
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        for epoch in range(self.rounds):
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                # logging.debug(
                #     "Starting minibatch {} with num_samples: {}".format(
                #         count, len(data)
                #     )
                # )
                # logging.debug("Classes: {}".format(target))
                epoch_loss += self.trainstep(data, target)
                count += 1
            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    def eval_loss(self, dataset):
        """
        Redefined, forgot to add the eval() call to the model
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.eval()  # that
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss

    def compute_per_sample_loss(self, dataset: Dataset, loss_func):
        """
        Compute the per sample loss for the current model (the one that will be shared).

        Args:
            dataset (decentralizepy.datasets.Dataset): The training dataset.
            loss_func: Loss function, must have reduction set to none.

        Returns:
            list: List containing the per sample loss
        """
        self.model.eval()
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        with torch.no_grad():
            per_sample_loss = []
            for data, target in trainset:
                output = self.model(data)
                losses = loss_func(output, target)
                per_sample_loss.extend(losses.tolist())
        return per_sample_loss
