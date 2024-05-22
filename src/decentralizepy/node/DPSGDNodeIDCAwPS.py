import importlib
import logging
import math
import os
from typing import Dict, List  # noqa: F401

import torch

from decentralizepy.communication.Communication import Communication  # noqa: F401
from decentralizepy.datasets.RotatedMNIST import RotatedMNIST  # noqa: F401
from decentralizepy.graphs.Graph import Graph  # noqa: F401
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model  # noqa: F401
from decentralizepy.node.DPSGDNodeIDCA import DPSGDNodeIDCA
from decentralizepy.sharing.CurrentModelSharing import CurrentModelSharing
from decentralizepy.sharing.Sharing import Sharing  # noqa: F401
from decentralizepy.training.TrainingIDCA import TrainingIDCA  # noqa: F401


class DPSGDNodeIDCAwPS(DPSGDNodeIDCA):
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
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
            [NODE]
                log_per_sample_loss = False
                log_per_sample_pred_true = False
                graph_package = decentralizepy.graphs.Regular
                graph_class = Regular
                graph_degree = 3
                graph_seed = 1234
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
        self.init_node(config["NODE"])
        self.init_dataset_models_parrallel(config["DATASET"])
        self.init_optimizer_config(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)  # could remove

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()  # could remove

    def init_node(self, node_config):
        """
        Initialize the node atribute and the graph object.

        Parameters
        ----------
        node_config : dict
            Configuration for the graph

        """
        self.log_per_sample_loss = node_config["log_per_sample_loss"]
        self.log_per_sample_pred_true = node_config["log_per_sample_pred_true"]
        self.do_all_reduce_models = node_config["do_all_reduce_models"]
        self.layers_sharing = node_config["layers_sharing"]

        graph_package = importlib.import_module(node_config["graph_package"])
        self.graph_class = getattr(graph_package, node_config["graph_class"])

        self.graph_degree = node_config["graph_degree"]
        self.graph_seed = node_config["graph_seed"]
        self.graph = self.graph_class(self.n_procs, self.graph_degree, self.graph_seed)  # type: Graph

    def get_neighbors(self, node=None):
        self.get_new_graph()
        return self.graph.neighbors(self.uid)

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

    def get_data_to_send(self) -> Dict:
        """Gets the data to send to neighbors.
        Decide if it should send it's current model or if it just has one.

        Returns:
            Dict: Data to send to neighbors
        """
        if self.sharing_class == CurrentModelSharing:
            # send only one model
            to_send = self.sharing.get_data_to_send(self.trainer.current_model_idx, len(self.my_neighbors))
        else:
            # send all model like in the non-Peer Sampler case
            to_send = self.sharing.get_data_to_send(len(self.my_neighbors))
        return to_send
