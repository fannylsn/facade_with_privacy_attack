import importlib
import json
import logging
import os
from collections import deque
from typing import Dict

from decentralizepy import utils
from decentralizepy.graphs.FullyConnected import FullyConnected
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.DPSGDWithPeerSampler import DPSGDWithPeerSampler
from decentralizepy.training.TrainingNIID import TrainingNIID  # noqa: F401
from decentralizepy.utils_learning_rates import get_lr_step_7_9


class DPSGDWithPeerSamplerNIID(DPSGDWithPeerSampler):
    """
    This class defines the node for DPSGD with peer sampler for non iid datasets.
    It just redifines the run method to log the cluster assigned to the node and some other methods to log metrics.

    """

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
            self.model,
            self.optimizer,
            self.loss,
            self.log_dir,
            **train_params,
        )  # type: TrainingNIID

    def init_node(self, node_config):
        """
        Initialize the node attribute.

        Args:
            node_config (dict): Configuration of the node
        """
        self.log_per_sample_loss = node_config["log_per_sample_loss"]
        self.log_per_sample_pred_true = node_config["log_per_sample_pred_true"]
        self.do_all_reduce_models = node_config["do_all_reduce_models"]
        self.graph_degree = node_config["graph_degree"]

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
            self.adjust_learning_rate(iteration)
            self.trainer.train(self.dataset)

            # sharing
            self.new_neighbors = self.get_neighbors()
            self.connect_neighbors()
            logging.debug("Connected to all neighbors")

            to_send = self.sharing.get_data_to_send(degree=len(self.my_neighbors))
            to_send["CHANNEL"] = "DPSGD"

            for neighbor in self.my_neighbors:
                self.communication.send(neighbor, to_send)

            while not self.received_from_all():
                sender, data = self.receive_DPSGD()
                logging.debug("Received Model from {} of iteration {}".format(sender, data["iteration"]))
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()

                if data["iteration"] == self.iteration:
                    self.peer_deques[sender].appendleft(data)
                else:
                    self.peer_deques[sender].append(data)

            averaging_deque = dict()
            for neighbor in self.my_neighbors:
                averaging_deque[neighbor] = self.peer_deques[neighbor]

            self.sharing._averaging(averaging_deque)

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

    def adjust_learning_rate(self, iteration: int):
        """Adjust the learning rate based on the iteration number.

        Args:
            iteration (int): current iteration

        """

        ratio = iteration / self.iterations
        new_params = self.optimizer_params.copy()
        new_params["lr"] = get_lr_step_7_9(ratio, new_params["lr"])
        logging.debug(f"learning rate: {new_params['lr']}")
        new_optimizer = self.optimizer_class(self.model.parameters(), **new_params)
        self.trainer.reset_optimizer(new_optimizer)

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
                "validation_loss": {},
                "validation_acc": {},
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
        Returns:
            Dict: dict containing the results
        """
        results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

        if hasattr(self.communication, "total_meta"):
            results_dict["total_meta"][str(iteration + 1)] = self.communication.total_meta
        if hasattr(self.communication, "total_data"):
            results_dict["total_data_per_n"][str(iteration + 1)] = self.communication.total_data
        return results_dict

    def write_results_dict(self, results_dict):
        """Dumps the results dictionary to a file.

        Args:
            results_dict (_type_): _description_
        """
        with open(os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w") as of:
            json.dump(results_dict, of)

    def compute_log_train_loss(self, results_dict, iteration):
        """Redefinition. Compute the train loss on the best model and save the plot.

        This is done after the averaging of models across neighboors.

        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        """
        if self.log_per_sample_loss:
            # log the per sample loss for MIA
            self.compute_log_per_sample_loss_train(results_dict, iteration)

        training_loss = self.trainer.eval_loss(self.dataset)

        results_dict["train_loss"][str(iteration + 1)] = training_loss

        self.save_plot(
            results_dict["train_loss"],
            "train_loss",
            "Training Loss",
            "Communication Rounds",
            os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
        )

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
        per_sample_loss_tr = self.trainer.compute_per_sample_loss(self.dataset, loss_func)
        results_dict["per_sample_loss_train"][str(iteration + 1)] = json.dumps(per_sample_loss_tr)
        return results_dict

    def eval_on_testset(self, results_dict: Dict, iteration):
        """Redefinition. Evaluate the model on the test set.
        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        Returns:
            dict: Dictionary containing the results
        """
        ta, tl = self.dataset.test(self.model, self.loss)
        results_dict["test_acc"][str(iteration + 1)] = ta
        results_dict["test_loss"][str(iteration + 1)] = tl

        # log some metrics for MIA and fairness
        self.compute_log_per_sample_metrics_test(results_dict, iteration)

        return results_dict

    def compute_log_per_sample_metrics_test(self, results_dict: Dict, iteration: int):
        """Compute the per sample metrics for the given model, if the flags are set.
        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
            best_idx (int): Index of the best model (previously computed)
        Returns:
            dict: Dictionary containing the results
        """
        loss_func = self.loss_class(reduction="none")

        if self.do_all_reduce_models:
            log_pred_this_iter = self.log_per_sample_pred_true and iteration == self.iterations
        else:
            log_pred_this_iter = self.log_per_sample_pred_true and iteration >= self.iterations - self.test_after

        per_sample_loss, per_sample_pred, per_sample_true = self.dataset.compute_per_sample_loss(
            self.model, loss_func, False, self.log_per_sample_loss, log_pred_this_iter
        )
        if self.log_per_sample_loss:
            results_dict["per_sample_loss_test"][str(iteration + 1)] = json.dumps(per_sample_loss)
        if log_pred_this_iter:
            results_dict["per_sample_pred_test"][str(iteration + 1)] = json.dumps(per_sample_pred)
            results_dict["per_sample_true_test"][str(iteration + 1)] = json.dumps(per_sample_true)
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

        va, vl = self.dataset.validate(self.model, self.loss)
        results_dict["validation_acc"][str(iteration + 1)] = va
        results_dict["validation_loss"][str(iteration + 1)] = vl
        return results_dict

    def compute_log_per_sample_loss_val(self, results_dict: Dict, iteration: int, best_idx: int):
        """Not used currently. Compute the per sample loss for the current model.

        Args:
            results_dict (dict): Dictionary containing the results
            iteration (int): current iteration
        Returns:
            dict: Dictionary containing the results
        """
        loss_func = self.loss_class(reduction="none")
        model = self.models[best_idx]
        per_sample_loss_val = self.dataset.compute_per_sample_loss(model, loss_func, validation=True)
        results_dict["per_sample_loss_val"][str(iteration + 1)] = json.dumps(per_sample_loss_val)
        return results_dict

    def all_reduce_model(self):
        """
        All reduce the model across all nodes.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be averaged

        Returns
        -------
        torch.nn.Module
            Averaged model

        """
        fc_graph = FullyConnected(self.mapping.get_n_procs())
        self.my_neighbors = fc_graph.neighbors(self.uid)
        self.connect_neighbors()

        to_send = self.sharing.get_data_to_send(degree=len(self.my_neighbors))
        to_send["CHANNEL"] = "DPSGD"

        for neighbor in self.my_neighbors:
            self.communication.send(neighbor, to_send)

        # fake a final iteration
        self.iteration = self.iterations
        while not self.received_from_all():
            sender, data = self.receive_DPSGD()
            logging.debug("Received Model from {} of iteration {}".format(sender, data["iteration"]))
            if sender not in self.peer_deques:
                self.peer_deques[sender] = deque()

            if data["iteration"] == self.iteration:
                self.peer_deques[sender].appendleft(data)
            else:
                self.peer_deques[sender].append(data)

        averaging_deque = dict()
        for neighbor in self.my_neighbors:
            averaging_deque[neighbor] = self.peer_deques[neighbor]

        self.sharing._averaging(averaging_deque)
