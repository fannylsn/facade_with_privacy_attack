import logging
from typing import List

from decentralizepy.graphs.DirectedGraph import DirectedGraph
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.PeerSamplerDynamic import PeerSamplerDynamic


class GraphOrchestratorDAC(PeerSamplerDynamic):
    """
    This class defines the peer sampling service

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
        log_level=logging.INFO,
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
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """

        self.iteration = 0

        self.received_counter = 0

        nodeConfigs = config["NODE"]
        self.graph_degree = nodeConfigs["graph_degree"]

        self.instantiate(
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
        self.graphs: List[DirectedGraph] = [DirectedGraph(self.graph.n_procs)]
        self.run()

        logging.info("Peer Sampler exiting")

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        log_level=logging.INFO,
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
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
        )

        self.init_dataset_model(config["DATASET"])

        self.message_queue = dict()

        self.barrier = set()

        self.init_comm(config["COMMUNICATION"])
        self.my_neighbors = self.graph.get_all_nodes()
        self.connect_neighbors()

    def run(self):
        """
        Start the peer-sampling service.

        """
        while len(self.barrier) > 0:
            sender, data = self.receive_server_request()
            pass  # sender ok ??
            if "BYE" in data:
                logging.debug("Received {} from {}".format("BYE", sender))
                self.barrier.remove(sender)

            elif "SEND_NEIGHBORS" in data:  # node sent its incomming neighbors
                logging.debug("Received {} from {}".format("SEND_NEIGHBORS", sender))
                if data["iteration"] != self.iteration:
                    raise ValueError("Iteration mismatch")

                self.received_counter += 1
                self.graphs[self.iteration].__update_incomming_edges__(
                    sender, data["SEND_NEIGHBORS"]
                )

            if self.received_counter == len(self.my_neighbors):
                # graph is full
                for neighbor in self.my_neighbors:
                    logging.debug("Sending neighbors to {}".format(neighbor))
                    resp = {
                        "NEIGHBORS": self.graphs[self.iteration].outgoing_edges(
                            neighbor
                        ),
                        "CHANNEL": "PEERS",
                    }

                    self.communication.send(neighbor, resp)

                # reset and prepare next iteration
                self.received_counter = 0
                self.graphs.append(
                    DirectedGraph(
                        self.graph.n_procs,
                    )
                )
                self.iteration += 1
