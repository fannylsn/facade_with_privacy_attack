import importlib
import json
import logging
import math
import os
from collections import deque
from typing import Dict, List  # noqa: F401

import numpy as np
import torch

from decentralizepy.communication.Communication import Communication  # noqa: F401
from decentralizepy.datasets.RotatedMNIST import RotatedMNIST  # noqa: F401
from decentralizepy.graphs.Graph import Graph  # noqa: F401
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model  # noqa: F401
from decentralizepy.node.DPSGDNodeIDCA import DPSGDNodeIDCA
from decentralizepy.sharing.Sharing import Sharing  # noqa: F401
from decentralizepy.training.TrainingIDCA import TrainingIDCA  # noqa: F401


class DPSGDNodeIDCAwPS(DPSGDNodeIDCA):
    """
    This class defines the node for DPSGD

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

        graph = None
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
        self.init_graph(config["GRAPH"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()

    def init_graph(self, graph_config):
        """
        Initialize the graph object.

        Parameters
        ----------
        graph_config : dict
            Configuration for the graph

        """
        graph_package = importlib.import_module(graph_config["graph_package"])
        self.graph_class = getattr(graph_package, graph_config["graph_class"])

        self.graph_degree = graph_config["graph_degree"]
        self.graph_seed = graph_config["graph_seed"]
        self.graph = self.graph_class(self.n_procs, self.graph_degree, self.graph_seed)  # type: Graph

    def get_neighbors(self, node=None):
        return self.graph.neighbors(self.uid)

    def get_new_graph(self, iteration):
        """
        Get the new graph for the current iteration

        Parameters
        ----------
        iteration : int
            Current iteration

        """
        self.graph = self.graph_class(
            self.n_procs,
            self.graph_degree,
            seed=self.graph_seed * 100000 + iteration,
        )
        pass

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

            self.get_new_graph(iteration)
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
