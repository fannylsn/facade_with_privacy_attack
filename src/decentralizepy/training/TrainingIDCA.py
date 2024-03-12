import logging
import random
from typing import List

import torch

from decentralizepy import utils
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.models.Model import Model
from decentralizepy.training.Training import Training


class TrainingIDCA(Training):
    """
    This class implements the training module for a single node of the IDCA algorithm.
    It recieves the n models, chooses the best one and trains it.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        models: List[Model],
        optimizer_class,
        optimizer_params,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        models : List[torch.nn.Module]
            List of Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs : bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.

        """
        self.models = models
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.loss = loss
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.rounds = utils.conditional_value(rounds, "", int(1))
        self.full_epochs = utils.conditional_value(full_epochs, "", False)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)

    def reset_optimizer(self, optimizer):
        """
        Replace the current optimizer with a new one

        Parameters
        ----------
        optimizer : torch.optim
            A new optimizer

        """
        self.optimizer = optimizer

    def eval_loss(self, model: Model, dataset: Dataset):
        """
        Evaluate the loss on the training set on the given model

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        model.eval()  # set the model to inference mode
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                output = model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss

    def trainstep(self, data, target):
        """
        One training step on a minibatch.

        Parameters
        ----------
        data : any
            Data item
        target : any
            Label

        Returns
        -------
        int
            Loss Value for the step

        """
        self.best_model.zero_grad()
        output = self.best_model(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def train_full(self, dataset):
        """
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        for epoch in range(self.rounds):
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                logging.debug(
                    "Starting minibatch {} with num_samples: {}".format(
                        count, len(data)
                    )
                )
                logging.debug("Classes: {}".format(target))
                epoch_loss += self.trainstep(data, target)
                count += 1
            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    def train(self, dataset: Dataset, treshold: float = 0.0):
        """
        One training iteration

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)
        treshold : float, optional
            Treshold in [0, 1] to explore the space. If set to 0, no exploration is done.

        """

        # chose the best model
        # self.choose_best_model(dataset)
        self.choose_best_model_explore(dataset, treshold=treshold)

        # reset the optimizer
        self.reset_optimizer(
            self.optimizer_class(self.best_model.parameters(), **self.optimizer_params)
        )

        self.best_model.train()  # set the best model to train mode

        if self.full_epochs:
            self.train_full(dataset)
        else:
            iter_loss = 0.0
            count = 0
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            while count < self.rounds:
                for data, target in trainset:
                    iter_loss += self.trainstep(data, target)
                    count += 1
                    logging.debug("Round: {} loss: {}".format(count, iter_loss / count))
                    if count >= self.rounds:
                        break

    def choose_best_model(self, dataset: Dataset):
        """
        Choose the best model from the list of models

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The dataset interface.
        """
        self.models_losses = [self.eval_loss(model, dataset) for model in self.models]
        self.best_model_loss = min(self.models_losses)
        self.best_model_idx = self.models_losses.index(self.best_model_loss)
        self.best_model = self.models[self.best_model_idx]

    def choose_best_model_explore(self, dataset: Dataset, treshold: float = 0.1):
        """
        Choose the best model from the list of models.
        Also, have a percentage of choosing a random model to explore the space.

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The dataset interface.
        """
        self.models_losses = [self.eval_loss(model, dataset) for model in self.models]
        if random.uniform(0, 1) < treshold:
            self.best_model_idx = random.randint(0, len(self.models) - 1)
            self.best_model_loss = self.models_losses[self.best_model_idx]
        else:
            self.best_model_loss = min(self.models_losses)
            self.best_model_idx = self.models_losses.index(self.best_model_loss)
        self.best_model = self.models[self.best_model_idx]

    def get_current_best_model(self) -> Model:
        """
        Get the best model

        Returns
        -------
        torch.nn.Module
            Best model

        """
        return self.best_model

    def get_best_model_loss(self) -> float:
        """
        Get the loss of the best model

        Returns
        -------
        float
            Loss of the best model

        """
        return self.best_model_loss

    def get_models_losses(self) -> List[float]:
        """
        Get the losses of all models

        Returns
        -------
        List[float]
            Losses of all models

        """
        return self.models_losses
