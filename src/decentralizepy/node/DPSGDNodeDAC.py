import copy
import importlib
import json
import logging
import math
import os
from collections import deque
from typing import Dict

import numpy as np
import torch

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.DPSGDWithPeerSamplerNIID import DPSGDWithPeerSamplerNIID
from decentralizepy.training.TrainingNIID import TrainingNIID  # noqa: F401


class DPSGDNodeDAC(DPSGDWithPeerSamplerNIID):
    """
    This class defines the node for DPSGD DISPFL with peer sampler for non iid datasets.
    It just redifines the run method to log the cluster assigned to the node and some other methods to log metrics.


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
        peer_sampler_uid=-1,
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
        self.threads_per_proc = max(math.floor(total_threads / mapping.get_local_procs_count()), 1)
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
        logging.info("Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads)

        self.message_queue["PEERS"] = deque()

        self.peer_sampler_uid = peer_sampler_uid
        self.connect_neighbor(self.peer_sampler_uid)
        self.wait_for_hello(self.peer_sampler_uid)

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
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])
        self.init_node(config["NODE"])

        self.message_queue = dict()

        self.barrier = set()

        # for DAC
        self.other_nodes = [i for i in range(self.graph.n_procs) if i != self.rank]
        self.one_over_loss = {idx: 0.0 for idx in self.other_nodes}
        self.prior_norm = {idx: 1.0 / (self.graph.n_procs - 1) for idx in self.other_nodes}

        self.tau_coef = 0.2
        self.tau_init = 30

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        # automatically done in run loop
        # self.connect_neighbors()

    def init_dataset_model(self, dataset_configs):
        """
        Redifined to keep the seed in self.
        Instantiate dataset and model from config.

        Parameters
        ----------
        dataset_configs : dict
            Python dict containing dataset config params

        """
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        self.dataset_class = getattr(dataset_module, dataset_configs["dataset_class"])
        random_seed = dataset_configs["random_seed"] if "random_seed" in dataset_configs else 97
        self.orig_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.dataset_params = utils.remove_keys(
            dataset_configs,
            ["dataset_package", "dataset_class", "model_class"],
        )
        self.dataset = self.dataset_class(self.rank, self.machine_id, self.mapping, **self.dataset_params)

        logging.info("Dataset instantiation complete.")

        self.model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.model = self.model_class()

    def run(self):
        """
        Start the decentralized learning.
        This method is a copy paste of the DPSGDWithPeerSampler run method with the
        addition of logging the cluster assigned to the node.

        """
        # rounds_to_test = self.test_after
        # rounds_to_train_evaluate = self.train_evaluate_after
        rounds_to_test = 1
        rounds_to_train_evaluate = 1

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1
            self.iteration = iteration

            # training
            # self.adjust_learning_rate(iteration)
            self.trainer.train(self.dataset)

            # sharing
            # get the neighboors we want to receive from
            self.my_incomming_neighbors = self.get_incomming_neighbors()
            logging.info("Incomming neighbors (to recieve from): {}".format(self.my_incomming_neighbors))
            # get the neighbors we have to send to
            self.my_outgoing_neighbors = self.get_outgoing_neighbors()
            logging.info("Outgoing neighbors (to send to): {}".format(self.my_outgoing_neighbors))
            self.my_neighbors = self.my_incomming_neighbors.union(self.my_outgoing_neighbors)
            self.connect_neighbors()
            logging.debug("Connected to all neighbors")

            to_send = self.sharing.get_data_to_send()
            to_send["CHANNEL"] = "DPSGD"

            for neighbor in self.my_outgoing_neighbors:
                self.communication.send(neighbor, to_send)

            while not self.received_from_all_incomming():
                sender, data = self.receive_DPSGD()
                logging.debug("Received Model from {} of iteration {}".format(sender, data["iteration"]))
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()

                if data["iteration"] == self.iteration:
                    self.peer_deques[sender].appendleft(data)
                else:
                    self.peer_deques[sender].append(data)

            averaging_deque = dict()
            for neighbor in self.my_incomming_neighbors:
                averaging_deque[neighbor] = self.peer_deques[neighbor]

            self.compute_neighbors_model_losses(averaging_deque)
            self.compute_prior_normalized()

            self.sharing._averaging(averaging_deque)

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            # logging and plotting
            results_dict = self.get_results_dict(iteration=iteration)
            results_dict = self.log_metadata(results_dict, iteration)

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after
                results_dict = self.compute_log_train_loss(results_dict, iteration)

            if rounds_to_test == 0:
                rounds_to_test = self.test_after

                if self.dataset.__testing__:
                    logging.info("evaluating on test set.")
                    results_dict = self.eval_on_testset(results_dict, iteration)

                if self.dataset.__validating__:
                    logging.info("evaluating on validation set.")
                    results_dict = self.eval_on_validationset(results_dict, iteration)

            self.write_results_dict(results_dict)

        # done with all iters
        if self.do_all_reduce_models:
            raise NotImplementedError("All reduce models not implemented for DPSGDNodeDAC (and should be of no use)")
            self.all_reduce_model()

            # final test
            results_dict = self.get_results_dict(iteration=self.iterations)
            results_dict = self.compute_log_train_loss(results_dict, self.iterations)
            results_dict = self.eval_on_testset(results_dict, self.iterations)
            results_dict = self.eval_on_validationset(results_dict, self.iterations)
            self.write_results_dict(results_dict)

            iteration = self.iterations

        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(self.log_dir, "{}_shared_parameters.json".format(self.rank)),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")

    def get_incomming_neighbors(self):
        # reseed to have different randomness in nodes
        np.random.seed(self.rank * self.orig_seed)

        # handle case where there are not enough non-zero probabilities
        non_zero_indices = [i for i, p in self.prior_norm.items() if p > 0]
        num_non_zero = len(non_zero_indices)
        if num_non_zero >= self.graph_degree:
            # If there are enough non-zero probabilities, sample directly
            neighbors = set(
                np.random.choice(self.other_nodes, self.graph_degree, replace=False, p=list(self.prior_norm.values()))
            )
        else:
            # If there are fewer non-zero probabilities than required samples,
            # sample all non-zero indices and fill the remaining slots with random samples
            remaining_slots = self.graph_degree - num_non_zero
            remaining_indices = set(np.random.choice(self.other_nodes, remaining_slots, replace=False))
            neighbors = set(non_zero_indices).union(remaining_indices)

        # put back old seed
        np.random.seed(self.orig_seed)
        return neighbors

    def get_outgoing_neighbors(self):
        # first send neighbors to graph orchestrator
        self.communication.send(
            self.peer_sampler_uid,
            {
                "CHANNEL": "SERVER_REQUEST",
                "SEND_NEIGHBORS": list(self.my_incomming_neighbors),
                "iteration": self.iteration,
            },
        )
        # then wait for the response
        _, data = self.receive_channel("PEERS")
        return set(data["NEIGHBORS"])

    def received_from_all_incomming(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_incomming_neighbors:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def compute_neighbors_model_losses(self, averaging_deque: Dict):
        neighbors_data = averaging_deque.copy()
        neigh_model = copy.deepcopy(self.model)
        for neigh_rank, neigh_deque in neighbors_data.items():
            neigh_state_dict = self.sharing.deserialized_model(neigh_deque[0])
            neigh_model.load_state_dict(neigh_state_dict)
            loss = self.dataset.get_model_loss_on_trainset(neigh_model, self.loss)
            self.one_over_loss[neigh_rank] = 1.0 / loss

    def compute_prior_normalized(self):
        tau = self.tau_function(self.iteration, self.tau_init, self.tau_coef)

        softmax = torch.nn.Softmax(dim=0)
        values = tau * torch.tensor(list(self.one_over_loss.values()), dtype=torch.float32)

        values = softmax(values)
        values = np.asarray(values).astype("float64")  # !!!!!
        values = values / np.sum(values)  # Compute the sum of values

        self.prior_norm = dict(zip(self.other_nodes, values.tolist()))

        logging.debug(self.one_over_loss)
        logging.debug(f"Prior probas: {self.prior_norm}")

    def tau_function(self, x, a, b):
        tau = 2 * a / (1 + np.exp(-b * x)) - a + 1
        return tau

    def make_probabilities(self, data) -> np.ndarray:
        strengths = np.array([prob for prob in data])
        return strengths / strengths.sum()
