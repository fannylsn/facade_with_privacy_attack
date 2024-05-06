import copy
import logging

import numpy as np
import torch

from decentralizepy import utils
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.training.Training import Training


class TrainingDisPFL(Training):
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
        dense_ratio=0.5,
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

        self.dense_ratio = dense_ratio

        self.params = self.get_trainable_params()
        self.sparsities = self.calculate_sparsities(self.params, self.dense_ratio)

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

    # _______________DISPFL_______________

    def get_trainable_params(self):
        dict = {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse=0.5, erk_power_scale=1.0):
        """???

        Args:
            params (_type_): _description_
            tabu (list, optional): _description_. Defaults to [].
            distribution (str, optional): _description_. Defaults to "ERK".
            sparse (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        spasities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - self.dense_ratio
                else:
                    spasities[name] = 0
        elif distribution == "ERK":
            logging.info("initialize by ERK")
            total_params = 0
            for name in params:
                total_params += params[name].numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()

            density = sparse
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                            np.sum(params[name].shape) / np.prod(params[name].shape)
                        ) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if name in dense_layers:
                    spasities[name] = 0
                else:
                    spasities[name] = 1 - epsilon * raw_probabilities[name]
        return spasities

    def init_masks(self, params, sparsities):
        masks = {}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] = 1
        return masks
