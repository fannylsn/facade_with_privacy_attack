import json
import logging
import os
from collections import deque

from decentralizepy.node.DPSGDWithPeerSampler import DPSGDWithPeerSampler


class DPSGDWithPeerSamplerNIID(DPSGDWithPeerSampler):
    """
    This class defines the node for DPSGD with peer sampler for non iid datasets.
    It just redifines the run method to log the cluster assigned to the node.

    """

    def run(self):
        """
        Start the decentralized learning.
        This method is a copy paste of the DPSGDWithPeerSampler run method with the
        addition of logging the cluster assigned to the node.

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
            self.trainer.train(self.dataset)

            new_neighbors = self.get_neighbors()

            # The following code does not work because TCP sockets are supposed to be long lived.
            # for neighbor in self.my_neighbors:
            #     if neighbor not in new_neighbors:
            #         logging.info("Removing neighbor {}".format(neighbor))
            #         if neighbor in self.peer_deques:
            #             assert len(self.peer_deques[neighbor]) == 0
            #             del self.peer_deques[neighbor]
            #         self.communication.destroy_connection(neighbor, linger = 10000)
            #         self.barrier.remove(neighbor)

            self.my_neighbors = new_neighbors
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

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][iteration + 1] = (
                    self.communication.total_meta
                )
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][iteration + 1] = (
                    self.communication.total_data
                )

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl
                if self.dataset.__validating__:
                    logging.info("Evaluating on the validation set")
                    va, vl = self.dataset.validate(self.model, self.loss)
                    results_dict["validation_acc"][iteration + 1] = va
                    results_dict["validation_loss"][iteration + 1] = vl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)
        if self.model.shared_parameters_counter is not None:
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
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")
