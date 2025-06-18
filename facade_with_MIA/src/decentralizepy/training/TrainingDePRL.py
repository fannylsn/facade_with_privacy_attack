import copy
import logging

import torch

from decentralizepy import utils
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.training.Training import Training


class TrainingDePRL(Training):
    """
    This class implements the training module for a DISPFL node.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
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
        model : torch.nn.Module
            Neural Network for training
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
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.rounds = utils.conditional_value(rounds, "", int(1))
        self.full_epochs = utils.conditional_value(full_epochs, "", False)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)

    def train(self, dataset):
        """
        One training iteration

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.train()
        # in their code, they do full epochs, in the paper they say to do gradient steps... inconsistent
        if self.full_epochs:
            # implements "way2" in DePRL paper
            # first train the head, then the bady

            # this is original, we tried to reduce to prevent head overfitting
            # self.train_full(dataset, self.rounds, train_head=True)
            self.train_full(dataset, 1, train_head=True)
            self.train_full(dataset, 1, train_head=False)
        else:
            self.train_partial(dataset, self.rounds, train_head=True)
            self.train_partial(dataset, 1, train_head=False)

    def train_full(self, dataset, rounds: int, train_head=False):
        """
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        params_before = copy.deepcopy(self.model.state_dict())
        for epoch in range(rounds):
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                # logging.debug("Starting minibatch {} with num_samples: {}".format(count, len(data)))
                # logging.debug("Classes: {}".format(target))
                epoch_loss += self.trainstep(data, target)
                count += 1

                # update only part of the model
                with torch.no_grad():
                    params_afer = copy.deepcopy(self.model.state_dict())
                    for k in self.model.state_dict().keys():
                        if train_head:
                            if not self.model.key_in_head(k):
                                # key is from body, keep old param
                                params_afer[k] = params_before[k]
                        else:
                            if self.model.key_in_head(k):
                                # key is from head, keep old param
                                params_afer[k] = params_before[k]
                    self.model.load_state_dict(params_afer)

            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    def train_partial(self, dataset, rounds: int, train_head=False):
        iter_loss = 0.0
        count = 0
        params_before = copy.deepcopy(self.model.state_dict())
        while count < rounds:
            # if we have not finished the rounds, we need to continue training with new ransom order
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            for data, target in trainset:
                iter_loss += self.trainstep(data, target)
                count += 1

                # update only part of the model
                with torch.no_grad():
                    params_afer = copy.deepcopy(self.model.state_dict())
                    for k in self.model.state_dict().keys():
                        if train_head:
                            if not self.model.key_in_head(k):
                                # key is from body, keep old param
                                params_afer[k] = params_before[k]
                        else:
                            if self.model.key_in_head(k):
                                # key is from head, keep old param
                                params_afer[k] = params_before[k]
                    self.model.load_state_dict(params_afer)

                logging.debug("Round: {} loss: {}".format(count, iter_loss / count))
                if count >= rounds:
                    break

    def eval_loss(self, dataset):
        """
        Redefined, forgot the eval() call
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.eval()
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
