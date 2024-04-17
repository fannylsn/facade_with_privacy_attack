import copy
import importlib
import json
import logging
import math
import os
import random
from collections import deque
from typing import List  # noqa: F401

import numpy as np
import torch

from decentralizepy import utils
from decentralizepy.communication.Communication import Communication  # noqa: F401
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model  # noqa: F401
from decentralizepy.node.Node import Node
from decentralizepy.sharing.IFCASharing import IFCASharing  # noqa: F401


class FederatedParameterServerIFCA(Node):
    """
    This class defines the parameter serving service for the IFCA algorithm.

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
        working_fraction=1.0,
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
        working_fraction : float
            Percentage of nodes participating in one global iteration
        args : optional
            Other arguments

        """
        # for early restarts
        self.return_dict = None
        if len(args) > 0:
            self.return_dict = args[0]

        graph = None
        super().__init__(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            log_level,
            *args,
        )

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
            *args,
        )

        self.working_fraction = working_fraction

        random.seed(self.mapping.get_uid(self.rank, self.machine_id))

        self.run()

        logging.info("Parameter Server exiting")

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
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
        )

        self.message_queue = dict()

        self.barrier = set()

        self.peer_deques = dict()

        self.init_models_parrallel(config["DATASET"])
        self.init_comm(config["COMMUNICATION"])

        self.connect_neighbors()

        self.init_sharing(config["SHARING"])

    def init_log(self, log_dir, log_level, force=True):
        """
        Instantiate Logging.

        Parameters
        ----------
        log_dir : str
            Logging directory
        rank : rank : int
            Rank of process local to the machine
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        force : bool
            Argument to logging.basicConfig()

        """
        log_file = os.path.join(log_dir, "ParameterServer.log")
        logging.basicConfig(
            filename=log_file,
            format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s",
            level=log_level,
            force=force,
        )

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

        """
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.n_procs = self.mapping.get_n_procs()
        self.my_neighbors = [i for i in range(self.n_procs)]
        self.log_dir = log_dir
        self.iterations = iterations
        self.sent_disconnections = False
        self.weights_store_dir = weights_store_dir
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after

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
            self.rank,
            self.machine_id,
            self.mapping,
            self.n_procs,
            **comm_params,
        )  # type: Communication

    def init_models_parrallel(self, dataset_configs):
        """
        Instantiate model from config. No dataset for server node.

        Parameters
        ----------
        dataset_configs : dict
            Python dict containing dataset config params

        """
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        random_seed = dataset_configs["random_seed"] if "random_seed" in dataset_configs else 97
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        # main seed, all nodes will receive those models
        self.model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.models = [self.model_class() for _ in range(dataset_configs["number_of_clusters"])]  # type: List[Model]

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
        sharing_params = utils.remove_keys(sharing_configs, ["sharing_package", "sharing_class"])
        self.sharing = self.sharing_class(
            self.rank,
            self.machine_id,
            self.communication,
            self.mapping,
            self.models,
            self.log_dir,
            **sharing_params,
        )  # type: IFCASharing

    def received_from_all(self):
        """
        Check if all current workers have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.current_workers:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def disconnect_neighbors(self):
        """
        Disconnects all neighbors.

        Raises
        ------
        RuntimeError
            If received another message while waiting for BYEs

        """
        if not self.sent_disconnections:
            logging.info("Disconnecting neighbors")

            for neighbor in self.my_neighbors:
                self.communication.send(neighbor, {"BYE": self.uid, "CHANNEL": "WORKER_REQUEST"})
                self.barrier.remove(neighbor)

            self.sent_disconnections = True

    def get_working_nodes(self):
        """
        Randomly select set of clients for the current iteration

        """
        k = int(math.ceil(len(self.my_neighbors) * self.working_fraction))
        return random.sample(self.my_neighbors, k)

    def run(self):
        """
        Start the federated parameter-serving service.

        """

        for iteration in range(self.iterations):
            self.iteration = iteration
            # reset deques after each iteration
            self.peer_deques = dict()

            # Get workers for this iteration
            # For IFCA, we currently take all workers -> working_fraction = 1
            self.current_workers = self.get_working_nodes()

            # Params to send to workers
            # need to do another class, server share all, node share 1
            to_send = self.sharing.get_data_to_send_server()
            to_send["CHANNEL"] = "WORKER_REQUEST"

            # Notify workers
            for worker in self.current_workers:
                self.communication.send(worker, to_send)

            # Receive updates from current workers
            while not self.received_from_all():
                sender, data = self.receive_channel("DPSGD")
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()

                if data["iteration"] == self.iteration:
                    self.peer_deques[sender].appendleft(data)
                else:
                    self.peer_deques[sender].append(data)

            logging.debug("Received from all current workers")

            # Average received updates
            averaging_deque = dict()
            for worker in self.current_workers:
                averaging_deque[worker] = self.peer_deques[worker]

            self.sharing._averaging_server(averaging_deque)

            # logging and plotting
            results_dict = self.get_results_dict(iteration=iteration)
            results_dict = self.log_metadata(results_dict, iteration)
            self.write_results_dict(results_dict)

            # check for early restart
            if self.return_dict is not None:
                if iteration != 0:
                    # check if some model is static -> means he will never be train again
                    for prev, curr in zip(self.prev_models, self.models):
                        static = True
                        for p1, p2 in zip(prev.parameters(), curr.parameters()):
                            if p1.data.ne(p2.data).sum() > 0:
                                static = False
                                break
                        if static:
                            self.return_dict["early_stop"] = True
                            self.disconnect_neighbors()
                            return
                self.prev_models = [copy.deepcopy(model) for model in self.models]

        self.disconnect_neighbors()
        logging.info("Storing final weight of each model")
        for i, model in enumerate(self.models):
            model.dump_weights(self.weights_store_dir, i, iteration)
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
                "total_bytes": {},
                "total_meta": {},
                "total_data_per_n": {},
            }
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
            results_dict["total_data_per_n"][iteration + 1] = self.communication.total_data
        return results_dict

    def write_results_dict(self, results_dict):
        """Dumps the results dictionary to a file.

        Args:
            results_dict (_type_): _description_
        """
        with open(os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w") as of:
            json.dump(results_dict, of)
