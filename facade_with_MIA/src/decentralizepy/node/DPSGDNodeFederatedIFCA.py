import importlib
import json
import logging
import math
import os
from collections import deque
from typing import Dict, List  # noqa: F401

import numpy as np
import torch
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.communication.Communication import Communication  # noqa: F401
from decentralizepy.datasets.RotatedDataset import RotatedDataset  # noqa: F401
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model  # noqa: F401
from decentralizepy.node.Node import Node
from decentralizepy.sharing.IFCASharing import IFCASharing  # noqa: F401
from decentralizepy.training.TrainingIDCA import TrainingIDCA  # noqa: F401
from decentralizepy.utils_learning_rates import get_lr_step_7_9


class DPSGDNodeFederatedIFCA(Node):
    """
    This class defines the node for federated DPSGD for IFCA algorithm.

    """

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        parameter_server_uid=-1,
        *args,
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
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        parameter_server_uid: int
            The parameter server's uid
        args : optional
            Other arguments

        """

        # for early restarts
        self.return_dict = None
        if len(args) > 0:
            self.return_dict = args[0]

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.get_local_procs_count()), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            *args,
        )
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )

        self.message_queue["PEERS"] = deque()

        self.parameter_server_uid = parameter_server_uid
        self.connect_neighbor(self.parameter_server_uid)
        self.wait_for_hello(self.parameter_server_uid)

        self.run()

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        *args,
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_models_parrallel(config["DATASET"])
        self.init_optimizer_config(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])
        self.init_node(config["NODE"])

        self.message_queue = dict()

        self.barrier = set()

        self.participated = 0

        self.init_sharing(config["SHARING"])

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.n_procs = self.mapping.get_n_procs()
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.n_procs, **comm_params
        )  # type: Communication

    def init_dataset_models_parrallel(self, dataset_configs):
        """
        Instantiate dataset and model from config. Also instantiate k models.

        Parameters
        ----------
        dataset_configs : dict
            Python dict containing dataset config params

        """
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        self.dataset_class = getattr(dataset_module, dataset_configs["dataset_class"])
        random_seed = (
            dataset_configs["random_seed"] if "random_seed" in dataset_configs else 97
        )
        torch.manual_seed(random_seed)
        np.random.seed(int(random_seed))
        self.dataset_params = utils.remove_keys(
            dataset_configs,
            ["dataset_package", "dataset_class", "model_class"],
        )
        self.dataset = self.dataset_class(
            self.rank, self.machine_id, self.mapping, **self.dataset_params
        )  # type: RotatedDataset

        logging.info("Dataset instantiation complete.")

        # The initialization of the models must be different for each node.
        # Actually, it doesn't matter as each node will receive the same model from the server.
        torch.manual_seed(random_seed * self.rank)
        np.random.seed(random_seed * self.rank)
        self.model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.models = [
            self.model_class() for _ in range(dataset_configs["number_of_clusters"])
        ]  # type: List[Model]

        # Put back the previous seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def init_optimizer_config(self, optimizer_configs):
        """
        Instantiate optimizer parameters from config.

        Parameters
        ----------
        optimizer_configs : dict
            Python dict containing optimizer config params

        """
        optimizer_module = importlib.import_module(
            optimizer_configs["optimizer_package"]
        )
        self.optimizer_class = getattr(
            optimizer_module, optimizer_configs["optimizer_class"]
        )
        self.original_optimizer_params = utils.remove_keys(
            optimizer_configs, ["optimizer_package", "optimizer_class"]
        )

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
            self.loss = self.loss_class()
        else:
            self.loss = getattr(loss_package, train_configs["loss"])

        train_params = utils.remove_keys(
            train_configs,
            [
                "training_package",
                "training_class",
                "loss",
                "loss_package",
                "loss_class",
            ],
        )
        self.trainer = train_class(
            self.rank,
            self.machine_id,
            self.mapping,
            self.models,
            self.optimizer_class,
            self.original_optimizer_params.copy(),
            self.loss,
            self.log_dir,
            **train_params,
        )  # type: TrainingIDCA

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
            self.models,
            self.log_dir,
            **sharing_params,
        )  # type: IFCASharing

    def init_node(self, node_config):
        """
        Initialize the node attribute.

        Args:
            node_config (dict): Configuration of the node
        """
        self.log_per_sample_loss = node_config["log_per_sample_loss"]
        self.log_per_sample_pred_true = node_config["log_per_sample_pred_true"]

    def run(self):
        """
        Start the decentralized learning

        """
        # rounds_to_test = self.test_after
        # rounds_to_train_evaluate = self.train_evaluate_after
        rounds_to_test = 1
        rounds_to_train_evaluate = 1

        while len(self.barrier):
            rounds_to_test -= 1
            rounds_to_train_evaluate -= 1

            # receive from server
            sender, data = self.receive_channel("WORKER_REQUEST")

            if "BYE" in data:
                logging.debug("Received {} from {}".format("BYE", sender))
                self.barrier.remove(sender)
                break

            # for early restarts
            if "RESTART" in data:
                logging.debug("Received {} from {}".format("RESTART", sender))
                self.barrier.remove(sender)
                self.return_dict["early_stop"] = True
                break

            iteration = data["iteration"]
            self.sharing.recieve_data_node(data)
            # after recieving, the current is not the best anymore
            self.trainer.current_model_is_best = False

            logging.debug(
                "Received worker request at node {}, global iteration {}, local round {}".format(
                    self.uid, iteration, self.participated
                )
            )

            # logging and ploting
            results_dict = self.get_results_dict()
            results_dict = self.log_metadata(results_dict, iteration)

            # testing
            # It must be done before training, as we want to test the models of
            # the central server on the local dataset.
            if rounds_to_test == 0:
                rounds_to_test = self.test_after

                if self.dataset.__testing__:
                    logging.info("evaluating on test set.")
                    results_dict = self.eval_on_testset(results_dict, iteration)

                if self.dataset.__validating__:
                    logging.info("evaluating on validation set.")
                    results_dict = self.eval_on_validationset(results_dict, iteration)

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after
                results_dict = self.compute_best_model_log_train_loss(
                    results_dict, iteration
                )

            self.write_results_dict(results_dict)

            # training
            logging.info("Starting training iteration")
            # No exploration by design in raw IFCA -> treshold = 0
            # self.adjust_learning_rate(iteration)
            self.trainer.train(self.dataset, 0)

            #######
            # with torch.no_grad():
            #     model = self.models[self.trainer.current_model_idx]
            #     new_state_dict = model.state_dict().copy()  # Create a copy of the state dict
            #     for k, v in new_state_dict.items():
            #         new_state_dict[k] = torch.ones_like(v) * self.rank
            #     model.load_state_dict(new_state_dict)  # Update the model's parameters
            #######
            # Send update to server
            to_send = self.sharing.get_data_to_send_node(self.trainer.current_model_idx)
            to_send["CHANNEL"] = "DPSGD"
            self.communication.send(self.parameter_server_uid, to_send)

            # update local iteration
            self.participated += 1

        logging.info("Server disconnected. Process complete!")

    def adjust_learning_rate(self, iteration: int):
        """Adjust the learning rate based on the iteration number.

        Args:
            iteration (int): current iteration

        """
        ratio = iteration / self.iterations
        new_params = self.original_optimizer_params.copy()
        new_params["lr"] = get_lr_step_7_9(ratio, new_params["lr"])
        logging.debug(f"learning rate: {new_params['lr']}")
        self.trainer.update_optimizer_params(
            new_params
        )  # only updates params of trainer, not node

    def get_results_dict(self):
        """Get the results dictionary, or create it."""
        if self.participated > 0:
            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                "r",
            ) as inf:
                results_dict = json.load(inf)
        else:
            results_dict = {
                "cluster_assigned": self.dataset.cluster,
                "train_loss": {},
                "all_train_loss": {str(idx): {} for idx in range(len(self.models))},
                "test_loss": {},
                "test_acc": {},
                "test_best_model_idx": {},
                "validation_loss": {},
                "validation_acc": {},
                "validation_best_model_idx": {},
                "total_bytes": {},
                "total_meta": {},
                "total_data_per_n": {},
            }
            if self.log_per_sample_loss:
                results_dict["per_sample_loss_test"] = {}
                results_dict["per_sample_loss_train"] = {}
            if self.log_per_sample_pred_true:
                results_dict["per_sample_pred_test"] = {}
                results_dict["per_sample_true_test"] = {}
        return results_dict

    def log_metadata(self, results_dict, iteration):
        """Log the metadata of the communication.

        Args:
            results_dict (Dict): dict containg the results
            iteration (int): current iteration
        """
        results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

        if hasattr(self.communication, "total_meta"):
            results_dict["total_meta"][iteration + 1] = self.communication.total_meta
        if hasattr(self.communication, "total_data"):
            results_dict["total_data_per_n"][
                iteration + 1
            ] = self.communication.total_data
        return results_dict

    def write_results_dict(self, results_dict):
        """Dumps the results dictionary to a file.

        Args:
            results_dict (_type_): _description_
        """
        with open(
            os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
        ) as of:
            json.dump(results_dict, of)

    def compute_best_model_log_train_loss(self, results_dict, iteration):
        """Compute the train loss on the best model and save the plot.

        This is done after the averaging of models across neighboors.

        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        """
        if not self.trainer.current_model_is_best:
            # If the current model is not the best, we need to choose the best model
            self.trainer.choose_best_model(self.dataset)

        if self.log_per_sample_loss:
            # log the per sample loss for MIA
            self.compute_log_per_sample_loss_train(results_dict, iteration)

        training_loss = self.trainer.get_current_model_loss()
        all_losses = self.trainer.get_all_models_loss()

        results_dict["train_loss"][iteration + 1] = training_loss
        for idx in range(len(self.models)):
            results_dict["all_train_loss"][str(idx)][iteration + 1] = all_losses[idx]

        self.save_plot(
            results_dict["train_loss"],
            "train_loss",
            "Training Loss",
            "Communication Rounds",
            os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
        )
        self.save_plot_models(results_dict["all_train_loss"])

        return results_dict

    def compute_log_per_sample_loss_train(self, results_dict: Dict, iteration: int):
        """Compute the per sample loss for the current model.
        Best model must be chosen before calling this function.
        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        Returns:
            dict: Dictionary containing the results
        """
        loss_func = self.loss_class(reduction="none")
        per_sample_loss_tr = self.trainer.compute_per_sample_loss(
            self.dataset, loss_func
        )
        results_dict["per_sample_loss_train"][str(iteration + 1)] = json.dumps(
            per_sample_loss_tr
        )
        return results_dict

    def save_plot(self, coords, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        coords : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [coords[key] for key in coords.keys()]
        x_axis = list(map(int, coords.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    def save_plot_models(self, models_losses: Dict[int, Dict]):
        """Save the plot of the models.

        Args:
            models_losses (Dict[int, Dict]): dict containing the losses of each model.
        """
        plt.clf()
        for idx, coords in models_losses.items():
            y_axis = list(iter(coords.values()))
            x_axis = list(map(int, coords.keys()))
            plt.plot(x_axis, y_axis, label=f"Model {idx}")
        plt.legend()
        plt.xlabel("Communication Rounds")
        plt.ylabel("Training Loss")
        plt.title("Training Loss of all models")
        plt.savefig(os.path.join(self.log_dir, f"{self.rank}_all_train_loss.png"))

    def eval_on_testset(self, results_dict: Dict, iteration):
        """Evaluate the model on the test set.

        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        """
        ta, tl, bidx = self.dataset.test(self.models, self.loss)
        results_dict["test_acc"][iteration + 1] = ta
        results_dict["test_loss"][iteration + 1] = tl
        results_dict["test_best_model_idx"][iteration + 1] = bidx

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
        Returns:
            dict: Dictionary containing the results
        """
        loss_func = self.loss_class(reduction="none")
        model = self.models[best_idx]

        # only log last iteration
        last_iteration = (
            self.iterations - 1 - (self.iterations - 1) % self.train_evaluate_after
        )
        log_pred_this_iter = (
            self.log_per_sample_pred_true and iteration == last_iteration
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
        """Evaluate the model on the validation set.

        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        """
        va, vl, _, bidx = self.dataset.validate(self.models, self.loss)
        results_dict["validation_acc"][iteration + 1] = va
        results_dict["validation_loss"][iteration + 1] = vl
        results_dict["validation_best_model_idx"][iteration + 1] = bidx
        return results_dict
