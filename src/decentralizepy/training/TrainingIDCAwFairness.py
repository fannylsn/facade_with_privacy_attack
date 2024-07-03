import logging
import random
from itertools import tee
from typing import List

import torch
import torch.utils
import torch.utils.data

from decentralizepy import utils
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.models.Model import Model
from decentralizepy.training.TrainingIDCA import TrainingIDCA
from decentralizepy.utils_decpy.losses import (
    FairLogLossLayerDiff,
    FairLogLossModelDiff,
    FeatureAlignmentLoss,
)


class TrainingIDCAwFairness(TrainingIDCA):
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
        losses,
        log_dir,
        fair_metric_dict,
        fair_metric_dict_other,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
        explore_models="",
        layers_sharing="",
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
        self.losses = losses
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.rounds = utils.conditional_value(rounds, "", int(1))
        self.full_epochs = utils.conditional_value(full_epochs, "", True)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)
        self.explore_models = utils.conditional_value(explore_models, "", False)
        self.layers_sharing = utils.conditional_value(layers_sharing, "", False)

        self.current_model_is_best = False

        self.fair_metric_dict = fair_metric_dict
        self.fair_metric_dict_other = fair_metric_dict_other

        self.loss_has_model = False
        if (
            isinstance(self.losses[0], FairLogLossModelDiff)
            or isinstance(self.losses[0], FairLogLossLayerDiff)
            or isinstance(self.losses[0], FeatureAlignmentLoss)
        ):
            self.loss_has_model = True
            for i, loss in enumerate(self.losses):
                loss.set_model(self.models[i])

    def train(self, dataset: Dataset, treshold: float = 0.0):
        """
        One training iteration

        Args:
            dataset (decentralizepy.datasets.Dataset): The training dataset. Should implement get_trainset(batch_size, shuffle)
            treshold (float, optional): Treshold in [0, 1] to explore the space. If set to 0, no exploration is done.
        """
        # update fair_metrics
        for idx, loss in enumerate(self.losses):
            if not self.loss_has_model:
                loss.set_fair_metric(self.fair_metric_dict[idx])
                loss.set_fair_metric_other(self.fair_metric_dict_other[idx])
            else:
                # mean of all models except idx
                with torch.no_grad():
                    all_other_models = [model for i, model in enumerate(self.models) if i != idx]
                    model_other = loss.compute_other_model(all_other_models)
                    # set in loss
                    loss.set_model_other(model_other)

        # logic not to redo the computation if the model is already the best
        if self.explore_models and random.uniform(0, 1) < treshold:
            # chosing a random model
            self.current_model_idx = random.randint(0, len(self.models) - 1)
            trainset_ori = dataset.get_trainset(self.batch_size, self.shuffle)  # all models eval on same samples
            trainsets = tee(trainset_ori, len(self.models))  # generator copy
            # self.models_losses = [self.eval_loss(model, trainset) for model, trainset in zip(self.models, trainsets)]
            self.models_losses = []
            for model, trainset, loss in zip(self.models, trainsets, self.losses):
                self.loss = loss
                self.loss.eval()
                self.models_losses.append(self.eval_loss(model, trainset))
            self.current_model_loss = self.models_losses[self.current_model_idx]
            self.current_model = self.models[self.current_model_idx]
            self.current_model_is_best = False
            logging.info(f"Random model chosen:{self.current_model_idx}")
        else:
            if not self.current_model_is_best:
                # If the current model is not the best, we need to choose the best model
                self.choose_best_model(dataset)

        model_leaking = False
        if model_leaking:
            a = 5.0
            self.leak_model(a)

        # pick correct loss to match the current model
        self.loss = self.losses[self.current_model_idx]

        # reset the optimizer to match the current model parameters
        self.reset_optimizer(self.optimizer_class(self.current_model.parameters(), **self.optimizer_params))

        self.current_model.train()  # set the current model to train mode
        self.loss.train()  # set the loss to train mode

        if self.full_epochs:
            self.train_full(dataset)
        else:
            self.train_partial_data(dataset)

        # update the fair metric after training
        self.fair_metric_dict[self.current_model_idx] = self.loss.get_fair_metric()
        self.loss.eval()  # set the loss to eval mode

        if self.layers_sharing:
            # share the layers of the trained model with other
            with torch.no_grad():
                layers = self.current_model.get_shared_layers()
                for model in self.models:
                    model.set_shared_layers(layers)

    def choose_best_model(self, dataset: Dataset):
        """
        Choose the best model from the list of models.
        Also, have a percentage of choosing a random model to explore the space.

        Args:
            dataset (decentralizepy.datasets.Dataset): The dataset interface.
            treshold (float, optional): Treshold in [0, 1] to explore the space. If set to 0, no exploration is done.
        """
        # chosing the best model
        trainset_ori = dataset.get_trainset(self.batch_size, self.shuffle)  # all models eval on same samples
        # trainset_ori = dataset.get_validationset(self.batch_size, self.shuffle)  # all models eval on same samples
        # logging.debug(f"using validation set of size {len(trainset_ori)}")
        trainsets = tee(trainset_ori, len(self.models))  # generator copy
        # self.models_losses = [self.eval_loss(model, trainset) for model, trainset in zip(self.models, trainsets)]
        self.models_losses = []
        for model, trainset, loss in zip(self.models, trainsets, self.losses):
            self.loss = loss
            self.loss.eval()
            self.models_losses.append(self.eval_loss(model, trainset))

        self.current_model_loss = min(self.models_losses)
        self.current_model_idx = self.models_losses.index(self.current_model_loss)
        self.current_model = self.models[self.current_model_idx]
        self.current_model_is_best = True
        logging.info(f"Best model chosen:{self.current_model_idx}")

    def eval_loss(self, model: Model, trainset: torch.utils.data.DataLoader):
        """
        Evaluate the loss on the training set on the given model

        Args:
            model (decentralizepy.models.Model): The model to evaluate.
            dataset (torch.utils.Dataloader): The training dataset.

        Returns:
            float: Loss value
        """
        model.eval()  # set the model to inference mode
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                logging.debug("Target: {}".format(target))
                output = model(data)
                if isinstance(self.loss, FeatureAlignmentLoss):
                    # forward pass for the other model to compute features
                    _ = self.loss.model_other(data)
                epoch_loss += self.loss(output, target).item()
                count += 1
                if not self.full_epochs:
                    if count >= self.rounds:
                        break
        loss = epoch_loss / count
        logging.info(f"Loss after {count} iteration: {loss}")
        return loss

    def _trainstep(self, data, target):
        """One training step on a minibatch.

        Args:
            data: Data item
            target: Label

        Returns:
            int: Loss Value for the step
        """
        # self.current_model.zero_grad()
        self.optimizer.zero_grad()
        output = self.current_model(data)
        if isinstance(self.loss, FeatureAlignmentLoss):
            # forward pass for the other model to compute features
            _ = self.loss.model_other(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        # logging.debug(
        #     f"gradient of cel:{self.loss.cross_entropy_loss.grad:.3f} and model diff:{self.loss.param_diff_mean.grad:.3f}"
        # )
        self.optimizer.step()
        return loss_val.item()

    def compute_per_sample_loss(self, dataset: Dataset, loss_func):
        """
        Compute the per sample loss for the current model (the one that will be shared).

        Args:
            dataset (decentralizepy.datasets.Dataset): The training dataset.
            loss_func: Loss function, must have reduction set to none.

        Returns:
            list: List containing the per sample loss
        """
        self.current_model.eval()
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        with torch.no_grad():
            per_sample_loss = []
            for data, target in trainset:
                output = self.current_model(data)
                if isinstance(loss_func, FeatureAlignmentLoss):
                    # forward pass for the other model to compute features
                    _ = loss_func.model_other(data)
                losses = loss_func(output, target)

                per_sample_loss.extend(losses.tolist())
        return per_sample_loss
