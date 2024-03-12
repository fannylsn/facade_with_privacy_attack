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
from decentralizepy.datasets.RotatedMNIST import RotatedMNIST  # noqa: F401
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model  # noqa: F401
from decentralizepy.node.Node import Node
from decentralizepy.sharing.Sharing import Sharing  # noqa: F401
from decentralizepy.training.TrainingIDCA import TrainingIDCA  # noqa: F401


class DPSGDNodeIDCA(Node):
    """
    This class defines the node for DPSGD

    """

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
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
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
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
        args : optional
            Other arguments

        """

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
            graph,
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
        self.run()

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
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
        graph : decentralizepy.graphs
            The object containing the global graph
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
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_models_parrallel(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
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
        graph : decentralizepy.graphs
            The object containing the global graph
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
        self.graph = graph
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
        Instantiate dataset and model from config.

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
        np.random.seed(random_seed)
        self.dataset_params = utils.remove_keys(
            dataset_configs,
            ["dataset_package", "dataset_class", "model_class"],
        )
        self.dataset = self.dataset_class(
            self.rank, self.machine_id, self.mapping, **self.dataset_params
        )  # type: RotatedMNIST

        logging.info("Dataset instantiation complete.")

        # The initialization of the models must be different for each node.
        torch.manual_seed(random_seed * self.rank)
        np.random.seed(random_seed * self.rank)
        self.model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.models = [
            self.model_class() for _ in range(dataset_configs["number_of_clusters"])
        ]  # type: List[Model]

        # Put back the previous seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def init_optimizer(self, optimizer_configs):
        """
        Instantiate optimizer from config.

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
        self.optimizer_params = utils.remove_keys(
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
            loss_class = getattr(loss_package, train_configs["loss_class"])
            self.loss = loss_class()
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
            self.optimizer_params,
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
        sharing_class = getattr(sharing_package, sharing_configs["sharing_class"])
        sharing_params = utils.remove_keys(
            sharing_configs, ["sharing_package", "sharing_class"]
        )
        self.sharing = sharing_class(
            self.rank,
            self.machine_id,
            self.communication,
            self.mapping,
            self.graph,
            self.models,
            self.dataset,
            self.log_dir,
            **sharing_params,
        )  # type: Sharing

    def get_neighbors(self, node=None):
        return self.my_neighbors

    def receive_DPSGD(self):
        return self.receive_channel("DPSGD")

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1
            self.iteration = iteration

            # best model choice in done in trainer
            treshold_explo = np.exp(-iteration * 3 / self.iterations)
            self.trainer.train(self.dataset, treshold_explo)

            self.my_neighbors = self.get_neighbors()
            self.connect_neighbors()
            logging.debug("Connected to all neighbors")

            to_send = self.sharing.get_data_to_send(degree=len(self.my_neighbors))
            to_send["CHANNEL"] = "DPSGD"

            for neighbor in self.my_neighbors:
                self.communication.send(neighbor, to_send)

            while not self.received_from_all():
                sender, data = self.receive_DPSGD()
                logging.debug(
                    "Received Model from {} of iteration {}".format(
                        sender, data["iteration"]
                    )
                )
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()

                if data["iteration"] == iteration:
                    self.peer_deques[sender].appendleft(data)
                else:
                    self.peer_deques[sender].append(data)

            averaging_deque = dict()
            for neighbor in self.my_neighbors:
                averaging_deque[neighbor] = self.peer_deques[neighbor]

            self.sharing._averaging(averaging_deque)

            results_dict = self.get_results_dict(iteration=iteration)

            results_dict = self.log_meta_to_results_dict(results_dict, iteration)

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                results_dict = self.compute_train_loss_after_avg(
                    results_dict, iteration
                )

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                results_dict = self.eval_on_testset(results_dict, iteration)
                if global_epoch == 49:
                    change *= 2
                global_epoch += change

                if self.dataset.__validating__:
                    logging.info("Evaluating on validation set.")
                    results_dict = self.eval_on_validationset(results_dict, iteration)

            self.write_results_dict(results_dict)

        # Done with all iterations
        final_best_model_idx = results_dict["test_best_model_idx"][self.iterations]
        final_best_model = self.models[final_best_model_idx]
        if final_best_model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        final_best_model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")

    def get_results_dict(self, iteration):
        """Get the results dictionary, or create it."""
        if iteration:
            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                "r",
            ) as inf:
                results_dict = json.load(inf)
        else:
            results_dict = {
                "cluster_assigned": self.dataset.cluster,
                "train_loss": {},
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
        return results_dict

    def log_meta_to_results_dict(self, results_dict, iteration):
        """_summary_

        Args:
            results_dict (Dict): dict containg the results
            iteration (int): current iteration
        """
        results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

        if hasattr(self.communication, "total_meta"):
            results_dict["total_meta"][iteration + 1] = self.communication.total_meta
        if hasattr(self.communication, "total_data"):
            results_dict["total_data_per_n"][iteration + 1] = (
                self.communication.total_data
            )
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

    def compute_train_loss_after_avg(self, results_dict, iteration):
        """Compute the train loss on the best model and save the plot.

        This is done after the averaging of models across neighboors.

        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        """
        self.trainer.choose_best_model(self.dataset)
        loss_after_sharing = self.trainer.get_best_model_loss()

        results_dict["train_loss"][iteration + 1] = loss_after_sharing
        self.save_plot(
            results_dict["train_loss"],
            "train_loss",
            "Training Loss",
            "Communication Rounds",
            os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
        )
        return results_dict

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
        return results_dict

    def eval_on_validationset(self, results_dict: Dict, iteration):
        """_summary_

        Args:
            results_dict (Dict): _description_
            iteration (_type_): _description_
        """
        va, vl, bidx = self.dataset.validate(self.models, self.loss)
        results_dict["validation_acc"][iteration + 1] = va
        results_dict["validation_loss"][iteration + 1] = vl
        results_dict["validation_best_model_idx"][iteration + 1] = bidx
        return results_dict

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_neighbors:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def save_plot(self, coords, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
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
