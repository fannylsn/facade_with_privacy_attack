import logging

import torch

from decentralizepy.sharing.CurrentModelSharing import CurrentModelSharing


class MuffliatoSharing(CurrentModelSharing):
    """
    API defining who to share with and what, and what to do on receiving

    """

    def get_data_to_send(
        self,
        model_idx: int,
        degree=None,
    ):
        self._pre_step()
        self.model = self.models[model_idx]
        data = self.serialized_model()
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        data["model_idx"] = model_idx
        data["degree"] = degree if degree != None else len(self.graph.neighbors(my_uid))
        data["iteration"] = self.communication_round
        return data

    def get_data_to_send_noisy(
        self,
        model_idx: int,
        degree=None,
        noise_std=0.0,
    ):
        self._pre_step()
        self.model = self.models[model_idx]

        # add noise
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] += torch.randn(state_dict[key].shape) * noise_std
        self.model.load_state_dict(state_dict)

        data = self.serialized_model()
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        data["model_idx"] = model_idx
        data["degree"] = degree if degree != None else len(self.graph.neighbors(my_uid))
        data["iteration"] = self.communication_round
        return data

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        Args:
            peer_deques: A dictionary of deques of received models from neighbors
        """
        with torch.no_grad():
            received_models = dict()
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                _, iteration, model_idx = (
                    data["degree"],
                    data["iteration"],
                    data["model_idx"],
                )
                del data["degree"]
                del data["iteration"]
                del data["model_idx"]
                del data["CHANNEL"]
                logging.debug("Averaging model from neighbor {} of iteration {}".format(n, iteration))
                data = self.deserialized_model(data)
                if model_idx in received_models:
                    received_models[model_idx].append(data)
                else:
                    received_models[model_idx] = [data]

            # iterate on all the current models of the node
            for idx, model in enumerate(self.models):
                total = {}
                # compute the weight based on the number of incomming models
                if idx in received_models:
                    weight = 1 / (len(received_models[idx]) + 1)
                else:
                    weight = 1.0
                # initialize the total model
                for key, value in model.state_dict().items():
                    total[key] = value * weight
                # add the received models
                if idx in received_models:
                    for rec_model in received_models[idx]:
                        for key, value in rec_model.items():
                            total[key] += value * weight
                # assign the new state to the model
                model.load_state_dict(total)

        self._post_step()
        self.communication_round += 1

    def _averaging_cheb(self, peer_deques, gamma):
        """
        Averages the received model with the local model.

        Implements the accelerated Chebyshev algorithm for averaging.

        Args:
            peer_deques: A dictionary of deques of received models from neighbors
            gamma: The Chebyshev acceleration parameter
        """
        with torch.no_grad():
            received_models = dict()
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                _, iteration, model_idx = (
                    data["degree"],
                    data["iteration"],
                    data["model_idx"],
                )
                del data["degree"]
                del data["iteration"]
                del data["model_idx"]
                del data["CHANNEL"]
                logging.debug("Averaging model from neighbor {} of iteration {}".format(n, iteration))
                data = self.deserialized_model(data)
                if model_idx in received_models:
                    received_models[model_idx].append(data)
                else:
                    received_models[model_idx] = [data]

            # iterate on all the current models of the node
            for idx, model in enumerate(self.models):
                total = {}
                # compute the weight based on the number of incomming models
                if idx in received_models:
                    weight = 1 / (len(received_models[idx]) + 1)
                else:
                    weight = 1.0
                # initialize the total model
                for key, value in model.state_dict().items():
                    total[key] = value * weight
                # add the received models
                if idx in received_models:
                    for rec_model in received_models[idx]:
                        for key, value in rec_model.items():
                            total[key] += value * weight
                # do the accelerated chebichev part
                raise NotImplementedError("Wrong implementation")
                for key, value in model.state_dict().items():
                    # value here should be the PREVIOUS model
                    total[key] = (1 - gamma) * value + gamma * total[key]
                # assign the new state to the model
                model.load_state_dict(total)

        self._post_step()
        self.communication_round += 1
