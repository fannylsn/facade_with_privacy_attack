import copy
import importlib
import json
import logging
import math
from typing import Dict, List  # noqa: F401

import torch

from decentralizepy import utils
from decentralizepy.communication.Communication import Communication  # noqa: F401
from decentralizepy.datasets.RotatedMNIST import RotatedMNIST  # noqa: F401
from decentralizepy.graphs.Graph import Graph  # noqa: F401
from decentralizepy.models.Model import Model  # noqa: F401
from decentralizepy.node.DPSGDNodeIDCAwPS import DPSGDNodeIDCAwPS
from decentralizepy.sharing.CurrentModelSharingFair import (
    CurrentModelSharingFair,
)  # noqa: F401
from decentralizepy.sharing.Sharing import Sharing  # noqa: F401
from decentralizepy.training.TrainingIDCAwFairness import (
    TrainingIDCAwFairness,
)  # noqa: F401
from decentralizepy.utils_decpy.losses import (  # noqa: F401
    FairLogLossDiffLoss,
    FairLogLossEquOdds,
    FairLogLossLayerDiff,
)


class DPSGDNodeIDCAwFairness(DPSGDNodeIDCAwPS):
    """
    This class defines the node for DPSGD with peer sampler for non iid datasets.
    Instead of having a fix graph topology, the graph is updated every iteration.

    """

    def init_trainer(self, train_configs):
        """
        Instantiate training module and loss from config.

        Parameters
        ----------
        train_configs : dict
            Python dict containing training config params

        """
        train_module = importlib.import_module(train_configs["training_package"])
        train_class = getattr(train_module, train_configs["training_class"])

        loss_package = importlib.import_module(train_configs["loss_package"])
        if "loss_class" in train_configs.keys():
            self.loss_class = getattr(loss_package, train_configs["loss_class"])
            lambda_ = train_configs.get("lambda_", 0.0)

            oracle = False
            if oracle:
                # if majority, put smaller value
                if self.dataset.cluster == 0:
                    lambda_ = 0.0
            logging.info(f"Using lambda value: {lambda_} (oracle: {oracle})")

            if self.loss_class == FairLogLossLayerDiff:
                num_blocks = len([block for block in self.models[0].get_layers()])
                self.losses = [
                    self.loss_class(lambda_=lambda_, num_blocks=num_blocks)
                    for _ in range(len(self.models))
                ]
            else:
                self.losses = [
                    self.loss_class(lambda_=lambda_) for _ in range(len(self.models))
                ]
            # self.loss = self.losses[0]
        else:
            raise ValueError("Loss class not found in config")

        train_params = utils.remove_keys(
            train_configs,
            [
                "training_package",
                "training_class",
                "loss",
                "loss_package",
                "loss_class",
                "lambda_",
            ],
        )

        # init fairness metrics
        self.n_class = getattr(self.dataset_module, "NUM_CLASSES")
        if self.loss_class == FairLogLossEquOdds:
            dim = 2 * self.n_class  # for tpr and fpr
            fill = 1 / self.n_class
        elif self.loss_class == FairLogLossDiffLoss:
            dim = 1  # just the loss
            fill = 0.0
        else:
            dim = self.n_class  # either acc or pos rate per class
            fill = 1 / self.n_class

        self.fair_metric_dict = {
            idx: torch.ones(dim).float() * fill for idx in range(len(self.models))
        }
        self.fair_metric_dict_other = {
            idx: torch.ones(dim).float() * fill for idx in range(len(self.models))
        }

        self.trainer = train_class(
            self.rank,
            self.machine_id,
            self.mapping,
            self.models,
            self.optimizer_class,
            self.original_optimizer_params.copy(),
            self.losses,  # DIFFERENCE
            self.log_dir,
            layers_sharing=self.layers_sharing,
            fair_metric_dict=self.fair_metric_dict,
            fair_metric_dict_other=self.fair_metric_dict_other,
            **train_params,
        )  # type: TrainingIDCAwFairness

    def init_sharing(self, sharing_configs):
        """
        Instantiate sharing module from config.

        Parameters
        ----------
        sharing_configs : dict
            Python dict containing sharing config params

        """
        sharing_package = importlib.import_module(sharing_configs["sharing_package"])
        self.sharing_class = getattr(sharing_package, sharing_configs["sharing_class"])
        sharing_params = utils.remove_keys(
            sharing_configs, ["sharing_package", "sharing_class"]
        )
        self.sharing = self.sharing_class(
            self.rank,
            self.machine_id,
            self.communication,
            self.mapping,
            self.graph,
            self.models,
            self.dataset,
            self.log_dir,
            fair_metric_dict=self.fair_metric_dict,
            fair_metric_dict_other=self.fair_metric_dict_other,
            layers_sharing=self.layers_sharing,
            **sharing_params,
        )  # type: CurrentModelSharingFair

    def get_data_to_send(self) -> Dict:
        """Gets the data to send to neighbors.
        Decide if it should send it's current model or if it just has one.

        Returns:
            Dict: Data to send to neighbors
        """
        current_idx = self.trainer.current_model_idx
        fair_metric = self.fair_metric_dict[current_idx]
        to_send = self.sharing.get_data_to_send(
            current_idx, fair_metric, len(self.my_neighbors)
        )
        return to_send

    def compute_log_per_sample_loss_train(self, results_dict: Dict, iteration: int):
        """Compute the per sample loss for the current model.
        Best model must be chosen before calling this function.
        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        Returns:
            dict: Dictionary containing the results
        """
        loss_func = self.loss_class(reduction="none")  # type: FairLogLossEquOdds
        loss_func.set_fair_metric(self.fair_metric_dict[self.trainer.current_model_idx])
        loss_func.set_fair_metric_other(
            self.fair_metric_dict_other[self.trainer.current_model_idx]
        )
        if self.trainer.loss_has_model:
            model = copy.deepcopy(self.models[self.trainer.current_model_idx])
            # model = self.models[self.trainer.current_model_idx].deepcopy()
            all_other_models = [
                model
                for i, model in enumerate(self.models)
                if i != self.trainer.current_model_idx
            ]
            other_model = loss_func.compute_other_model(all_other_models)
            loss_func.set_model(model)
            loss_func.set_model_other(other_model)

        per_sample_loss_tr = self.trainer.compute_per_sample_loss(
            self.dataset, loss_func
        )
        results_dict["per_sample_loss_train"][str(iteration + 1)] = json.dumps(
            per_sample_loss_tr
        )
        return results_dict

    def eval_on_testset(self, results_dict: Dict, iteration):
        """Redefinition. Evaluate the model on the test set.
        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        Returns:
            dict: Dictionary containing the results
        """
        for j, loss in enumerate(self.losses):
            loss.eval()
            if self.trainer.loss_has_model:
                # update other model (trained one might not be up to date)
                all_other_models = [
                    model for i, model in enumerate(self.models) if i != j
                ]
                other_model = loss.compute_other_model(all_other_models)
                loss.set_model_other(other_model)
        ta, tl, bidx = self.dataset.test(self.models, self.losses)  # give list of loss
        results_dict["test_acc"][str(iteration + 1)] = ta
        results_dict["test_loss"][str(iteration + 1)] = tl
        results_dict["test_best_model_idx"][str(iteration + 1)] = bidx

        # log some metrics for MIA and fairness
        self.compute_log_per_sample_metrics_test(results_dict, iteration, bidx)

        return results_dict

    def compute_log_per_sample_metrics_test(
        self, results_dict: Dict, iteration: int, best_idx: int
    ):
        """Compute the per sample metrics for the given model, if the flags are set.
        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
            best_idx (int): Index of the best model (previously computed)
        Returns:
            dict: Dictionary containing the results
        """
        loss_func = self.loss_class(reduction="none")  # type: FairLogLossEquOdds
        loss_func.set_fair_metric(self.fair_metric_dict[best_idx])
        loss_func.set_fair_metric_other(self.fair_metric_dict_other[best_idx])
        model = self.models[best_idx].deepcopy()
        if self.trainer.loss_has_model:
            all_other_models = [
                model for i, model in enumerate(self.models) if i != best_idx
            ]
            other_model = loss_func.compute_other_model(all_other_models)
            loss_func.set_model(model)
            loss_func.set_model_other(other_model)

        if self.do_all_reduce_models:
            log_pred_this_iter = (
                self.log_per_sample_pred_true and iteration == self.iterations
            )
        else:
            log_pred_this_iter = (
                self.log_per_sample_pred_true and iteration == self.iterations - 1
            )

        (
            per_sample_loss,
            per_sample_pred,
            per_sample_true,
        ) = self.dataset.compute_per_sample_loss(
            model, loss_func, False, self.log_per_sample_loss, log_pred_this_iter
        )
        if self.log_per_sample_loss:
            results_dict["per_sample_loss_test"][str(iteration + 1)] = json.dumps(
                per_sample_loss
            )
        if log_pred_this_iter:
            results_dict["per_sample_pred_test"][str(iteration + 1)] = json.dumps(
                per_sample_pred
            )
            results_dict["per_sample_true_test"][str(iteration + 1)] = json.dumps(
                per_sample_true
            )
        return results_dict

    def eval_on_validationset(self, results_dict: Dict, iteration):
        """Redefinition. Evaluate the model on the validation set.
        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        Returns:
            dict: Dictionary containing the results
        """
        # log the per sample loss for MIA, or don't
        # self.compute_log_per_sample_loss_val(results_dict, iteration)

        for loss in self.losses:
            loss.eval()
        va, vl, all_val_loss, bidx = self.dataset.validate(
            self.models, self.losses
        )  # losses

        for idx in range(len(self.models)):
            results_dict["all_val_loss"][str(idx)][str(iteration + 1)] = all_val_loss[
                idx
            ]
        self.save_plot_models(results_dict["all_val_loss"], "validation")

        results_dict["validation_acc"][str(iteration + 1)] = va
        results_dict["validation_loss"][str(iteration + 1)] = vl
        results_dict["validation_best_model_idx"][str(iteration + 1)] = bidx
        return results_dict

    def compute_log_per_sample_loss_val(
        self, results_dict: Dict, iteration: int, best_idx: int
    ):
        """Not used currently. Compute the per sample loss for the current model.

        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        Returns:
            dict: Dictionary containing the results
        """
        loss_func = self.loss_class(reduction="none")  # type: FairLogLossEquOdds
        loss_func.set_fair_metric(self.fair_metric_dict[best_idx])
        loss_func.set_fair_metric_other(self.fair_metric_dict_other[best_idx])
        if self.trainer.loss_has_model:
            model = copy.deepcopy(self.models[best_idx])
            all_other_models = [
                model for i, model in enumerate(self.models) if i != best_idx
            ]
            other_model = loss_func.compute_other_model(all_other_models)
            loss_func.set_model(model)
            loss_func.set_model_other(other_model)
        model = self.models[best_idx]
        per_sample_loss_val = self.dataset.compute_per_sample_loss(
            model, loss_func, validation=True
        )
        results_dict["per_sample_loss_val"][str(iteration + 1)] = json.dumps(
            per_sample_loss_val
        )
        return results_dict

    def after_eval_step(self, result_dict):
        use_val_to_train_ratio = False
        if use_val_to_train_ratio and self.loss_class == FairLogLossLayerDiff:
            training_loss = result_dict["train_loss"][str(self.iteration + 1)]
            validation_loss = result_dict["validation_loss"][str(self.iteration + 1)]
            val_over_train = validation_loss / training_loss
            scaler = abs(math.exp(val_over_train - 1) - 1)
            logging.info(
                f"Scaler at round {self.iteration}: {scaler}, train: {training_loss}, and val_over_train: {val_over_train}"
            )
            for loss in self.losses:
                loss.scale_lambda(scaler)
