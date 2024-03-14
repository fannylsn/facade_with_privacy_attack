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
        explore_models="",
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
        self.full_epochs = utils.conditional_value(full_epochs, "", True)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)
        self.explore_models = utils.conditional_value(explore_models, "", False)

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
        self.choose_best_model(dataset, treshold=treshold)

        # reset the optimizer to match the current model parameters
        self.reset_optimizer(
            self.optimizer_class(
                self.current_model.parameters(), **self.optimizer_params
            )
        )

        self.current_model.train()  # set the current model to train mode

        if self.full_epochs:
            self.train_full(dataset)
        else:
            self.train_partial_data(dataset)

    def choose_best_model(self, dataset: Dataset, treshold: float = 0.0):
        """
        Choose the best model from the list of models.
        Also, have a percentage of choosing a random model to explore the space.

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The dataset interface.
        """
        if self.explore_models:
            if random.uniform(0, 1) < treshold:
                # chosing a random model
                self.current_model_idx = random.randint(0, len(self.models) - 1)
                self.current_model_loss = None
                self.models_losses = None
                self.current_model = self.models[self.current_model_idx]
                self.current_model_is_best = False
                logging.info(f"Random model chosen:{self.current_model_idx}")
                return
        # chosing the best model
        self.models_losses = [self.eval_loss(model, dataset) for model in self.models]
        self.current_model_loss = min(self.models_losses)
        self.current_model_idx = self.models_losses.index(self.current_model_loss)
        self.current_model = self.models[self.current_model_idx]
        self.current_model_is_best = True
        logging.info(f"Best model chosen:{self.current_model_idx}")

    def train_full(self, dataset: Dataset):
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
                epoch_loss += self._trainstep(data, target)
                count += 1
            logging.debug("Epoch: {} mean loss: {}".format(epoch, epoch_loss / count))

    def train_partial_data(self, dataset: Dataset):
        """Probably useless, don't use it. Trains on a subset of the data.

        Args:
            dataset (Dataset): the dataset holding the data
        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        for epoch in range(self.rounds):
            count = 0
            iter_loss = 0.0
            for data, target in trainset:
                iter_loss += self._trainstep(data, target)
                count += 1
                logging.debug(
                    "Minibatch: {}, total mean loss: {}".format(
                        count, iter_loss / count
                    )
                )
                if count >= self.rounds:
                    logging.debug(f"Finished epoch {epoch} after {count} minibatchs")
                    count = 0
                    break

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

    def _trainstep(self, data, target):
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
        self.current_model.zero_grad()
        output = self.current_model(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def get_current_best_model(self) -> Model:
        """
        Get the best model

        Returns
        -------
        torch.nn.Module
            Best model

        """
        return self.current_model

    def get_current_model_loss(self) -> float:
        """
        Get the loss of the best model

        Returns
        -------
        float
            Loss of the best model

        """
        assert self.current_model_loss is not None
        return self.current_model_loss

    def get_all_models_loss(self) -> List[float]:
        """
        Get the losses of all models

        Returns
        -------
        List[float]
            Losses of all models

        """
        assert self.models_losses is not None
        return self.models_losses
