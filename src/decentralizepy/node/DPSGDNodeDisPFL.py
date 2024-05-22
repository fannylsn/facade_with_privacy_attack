import copy
import importlib
import json
import logging
import os
from collections import deque

import numpy as np
import torch

from decentralizepy import utils
from decentralizepy.node.DPSGDWithPeerSamplerNIID import DPSGDWithPeerSamplerNIID
from decentralizepy.training.TrainingDisPFL import TrainingDisPFL  # noqa: F401
from decentralizepy.training.TrainingNIID import TrainingNIID  # noqa: F401


class DPSGDNodeDisPFL(DPSGDWithPeerSamplerNIID):
    """
    This class defines the node for DPSGD DISPFL with peer sampler for non iid datasets.
    It just redifines the run method to log the cluster assigned to the node and some other methods to log metrics.

    """

    def init_node(self, node_config):
        """
        Initialize the node attribute.

        Args:
            node_config (dict): Configuration of the node
        """
        self.log_per_sample_loss = node_config["log_per_sample_loss"]
        self.log_per_sample_pred_true = node_config["log_per_sample_pred_true"]
        self.do_all_reduce_models = node_config["do_all_reduce_models"]
        self.cs = "random"  # TODO ??? mybe rel to graph

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
        )  # type: TrainingDisPFL

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

        self.model_params = self.trainer.get_model_params()  # BAD dont take copy
        self.mask = self.trainer.get_mask(self.model_params, self.trainer.sparsities)
        self.updates_matrix = copy.deepcopy(self.model_params)
        # mask some model weights
        for name in self.mask:
            self.model_params[name] = self.model_params[name] * self.mask[name]
            self.updates_matrix[name] = self.updates_matrix[name] - self.updates_matrix[name]

        # TODO ???
        self.dist_locals = np.zeros(shape=(self.n_procs, self.n_procs))

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1
            self.iteration = iteration

            # those are a list of all client in original disp
            model_param_last_round = copy.deepcopy(self.model_params)
            mask_pers_shared_last_round = copy.deepcopy(self.mask)

            # TODO ???
            self.dist_locals[self.uid][self.uid], total_dis = self.hamming_distance(
                mask_pers_shared_last_round, self.mask
            )
            logging.info("local mask changes: {} / {}".format(self.dist_locals[self.uid][self.uid], total_dis))

            # TODO prob useless we have peersampler
            nei_indexs = self._benefit_choose(
                iter,
                self.n_procs,
                self.n_procs,
                self.cs,
            )
            nei_indexs = np.sort(nei_indexs)

            # FILL DISTLOCAL MATRIX -> hamming dist between client masks do in orchestr

            # now request orchestrator to get the dists
            nei_distances = [self.dist_locals[self.uid][i] for i in nei_indexs]

            # new_model_masked, new_model = self._aggregate_func(
            #     clnt_idx,
            #     self.args.client_num_in_total,
            #     self.args.client_num_per_round,
            #     nei_indexs,
            #     w_per_mdls_lstrd,
            #     mask_pers_local,
            #     mask_pers_shared_lstrd,
            # )

            # share mask now ??
            # share model now ?? share not masked

            # test now (after agregating)

            # train with new model (overright self.model)
            # training
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

    def hamming_distance(self, mask_a, mask_b):
        dis = 0
        total = 0
        for key in mask_a:
            dis += torch.sum(mask_a[key].int() ^ mask_b[key].int())
            total += mask_a[key].numel()
        return dis, total

    def _benefit_choose(
        self,
        cur_clnt,
        client_num_in_total,
        client_num_per_round,
        cs=False,
        active_ths_rnd=1,
    ):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd == 1)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes == cur_clnt)[0]))
        return client_indexes
