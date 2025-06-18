import importlib
import json
import logging
import math
import os
from collections import deque
from typing import Dict, List  # noqa: F401

import networkx as nx
import numpy as np
import torch
from scipy.linalg import eigh

from decentralizepy.graphs.Graph import Graph  # noqa: F401
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.DPSGDNodeIDCAwPS import DPSGDNodeIDCAwPS
from decentralizepy.sharing.MuffliatoSharing import MuffliatoSharing


class DPSGDNodeIDCAwMUFF(DPSGDNodeIDCAwPS):
    """
    This class defines the node for DPSGD with peer sampler for non iid datasets.
    Instead of having a fix graph topology, the graph is updated every iteration.

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
            [NODE]
                log_per_sample_loss = True
                log_per_sample_pred_true = False
                graph_package = decentralizepy.graphs.Regular
                graph_class = Regular
                graph_degree = 3
                graph_seed = 1234
                noise_std = 0.01
                sharing_rounds_muffliato = 3
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

    def init_node(self, node_config):
        """
        Initialize the node object.

        Args:
            node_config (dict): Configuration of the node
        """
        self.sharing_rounds_muffliato = node_config["sharing_rounds_muffliato"]
        self.noise_std = node_config["noise_std"]

        self.log_per_sample_loss = node_config["log_per_sample_loss"]
        self.log_per_sample_pred_true = node_config["log_per_sample_pred_true"]

        graph_package = importlib.import_module(node_config["graph_package"])
        self.graph_class = getattr(graph_package, node_config["graph_class"])

        self.graph_degree = node_config["graph_degree"]
        self.graph_seed = node_config["graph_seed"]
        self.graph = self.graph_class(
            self.n_procs, self.graph_degree, self.graph_seed
        )  # type: Graph

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        # rounds_to_test = self.test_after
        # rounds_to_train_evaluate = self.train_evaluate_after
        rounds_to_test = 1
        rounds_to_train_evaluate = 1

        training_iteration = 0  # to keep track of the training iterations
        for iteration in range(self.iterations * self.sharing_rounds_muffliato):
            self.iteration = iteration

            if iteration % self.sharing_rounds_muffliato == 0:
                # every sharing_rounds_muffliato, we train
                logging.info("Starting training iteration: %d", training_iteration + 1)

                # best model choice in done in trainer
                treshold_explo = np.exp(-training_iteration * 3 / self.iterations)
                self.trainer.train(self.dataset, treshold_explo)

                # sharing noisy version
                self.share_receive_avg(share_noisy=True, make_new_graph=True)
            else:
                # rest of the time, we just share
                self.share_receive_avg(share_noisy=False, make_new_graph=False)

            # logging and plotting
            if (
                iteration % self.sharing_rounds_muffliato
                == self.sharing_rounds_muffliato - 1
            ):
                # last iteration of the sharing round
                results_dict = self.get_results_dict(iteration=training_iteration)
                results_dict = self.log_metadata(results_dict, training_iteration)
                rounds_to_train_evaluate -= 1
                rounds_to_test -= 1

                if rounds_to_train_evaluate == 0:
                    logging.info("Evaluating on train set.")
                    rounds_to_train_evaluate = self.train_evaluate_after
                    results_dict = self.compute_best_model_log_train_loss(
                        results_dict, training_iteration
                    )

                if rounds_to_test == 0:
                    rounds_to_test = self.test_after

                    if self.dataset.__testing__:
                        logging.info("evaluating on test set.")
                        results_dict = self.eval_on_testset(
                            results_dict, training_iteration
                        )

                    if self.dataset.__validating__:
                        logging.info("evaluating on validation set.")
                        results_dict = self.eval_on_validationset(
                            results_dict, training_iteration
                        )

                self.write_results_dict(results_dict)
                # update training iter after the whole sharing done for the step
                training_iteration += 1

        # Done with all iterations
        last_training_iteration = (
            self.iterations - (self.iterations - 1) % self.train_evaluate_after
        )

        final_best_model_idx = results_dict["test_best_model_idx"][
            str(last_training_iteration)
        ]
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
        final_best_model.dump_weights(
            self.weights_store_dir, self.uid, training_iteration
        )
        logging.info("All neighbors disconnected. Process complete!")

    def share_receive_avg(self, share_noisy: bool = False, make_new_graph: bool = True):
        """Share the model with neighbors, receive models from neighbors and average them.

        Args:
            share_noisy (bool, optional): Whether to share noisy model. Defaults to False.
            make_new_graph (bool, optional): Whether to make a new graph. Defaults to True.
        """
        if make_new_graph:
            self.my_neighbors = self.get_neighbors()
        else:
            self.my_neighbors = self.my_neighbors
        self.connect_neighbors()
        logging.debug("Connected to all neighbors")

        if share_noisy:
            # not_noisy_model = self.model.copy() NO ?

            to_send = self.sharing.get_data_to_send_noisy(  # uses self.models
                self.trainer.current_model_idx, len(self.my_neighbors), self.noise_std
            )
            # put back the real model for the local node : -> NO?
            # self.model = not_noisy_model
        else:
            to_send = self.get_data_to_send()

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

            if data["iteration"] == self.iteration:
                self.peer_deques[sender].appendleft(data)
            else:
                self.peer_deques[sender].append(data)

        averaging_deque = dict()
        for neighbor in self.my_neighbors:
            averaging_deque[neighbor] = self.peer_deques[neighbor]

        self.sharing._averaging_cheb(averaging_deque, self.gamma)  # with acc. chebychev
        # after averaging, the current is not the best anymore
        self.trainer.current_model_is_best = False

    def get_new_graph(self):
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
            seed=self.graph_seed * 100000 + self.iteration,
        )
        self.lambda_2 = self.vplambda(self.graph)
        self.gamma = self.gamma_tcheb(self.lambda_2)

    def vplambda(self, graph):
        """Compute the spectral gap of a networkx graph
        Args:
            graph (decentralizedpy.graphs.Graph): The graph
        """
        # compute the W expected # warning, not sure if work with IDCA (graph~model_idx)
        adj_list = {x: list(adj.union({x})) for x, adj in enumerate(graph.adj_list)}
        self_loop_graph = nx.Graph(adj_list)
        W = nx.to_numpy_array(self_loop_graph) / (self.graph_degree + 1)
        eigen = eigh(
            np.eye(self.n_procs) - W, eigvals_only=True, subset_by_index=[0, 1]
        )
        lambda_2 = eigen[1]
        assert 0 < lambda_2 < 1
        return lambda_2

    def gamma_tcheb(self, lambda2):
        """Compute the factor gamma in the tchebychev polynomials for the acceleration

        Args:
            lambda2 (float): Spectral gap of the graph, second smallest eigenvalue of the laplacian ????
        Returns:
            float: The factor gamma
        """
        return 2 * (1 - np.sqrt(lambda2 * (1 - lambda2 / 4))) / (1 - lambda2 / 2) ** 2

    def get_data_to_send(self) -> Dict:
        """Gets the data to send to neighbors.
        Decide if it should send it's current model or if it just has one.

        Returns:
            Dict: Data to send to neighbors
        """
        if self.sharing_class == MuffliatoSharing:
            # send only one model
            to_send = self.sharing.get_data_to_send(
                self.trainer.current_model_idx, len(self.my_neighbors)
            )
        else:
            raise NotImplementedError("Implement other sharing classes")
        return to_send
