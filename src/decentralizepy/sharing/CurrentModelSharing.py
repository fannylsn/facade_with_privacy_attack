import copy
import importlib
import logging
from typing import List

import torch

from decentralizepy.models.Model import Model
from decentralizepy.sharing.Sharing import Sharing


class CurrentModelSharing(Sharing):
    """
    API defining who to share with and what, and what to do on receiving

    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        models: List[Model],
        dataset,
        log_dir,
        layers_sharing=False,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph reprensenting neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data. Not implemented yet! TODO
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)

        """
        self.rank = rank
        self.machine_id = machine_id
        self.uid = mapping.get_uid(rank, machine_id)
        self.communication = communication
        self.mapping = mapping
        self.graph = graph
        self.models = models
        self.number_of_models = len(models)
        self.model = None
        self.dataset = dataset
        self.communication_round = 0
        self.log_dir = log_dir

        self.layers_sharing = layers_sharing

        self.shapes = []
        self.lens = []
        with torch.no_grad():
            for _, v in self.models[0].state_dict().items():
                self.shapes.append(v.shape)
                t = v.flatten().numpy()
                self.lens.append(t.shape[0])

        self.compress = compress

        if compression_package and compression_class:
            compressor_module = importlib.import_module(compression_package)
            compressor_class = getattr(compressor_module, compression_class)
            self.compressor = compressor_class(float_precision=float_precision)
            logging.debug(f"Using the {compressor_class} to compress the data")
        else:
            assert not self.compress

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

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

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

            if self.layers_sharing:
                # average of the common layers
                all_recieved = []
                for models in received_models.values():
                    all_recieved.extend(models)
                weight = 1 / (len(all_recieved) + 1)
                # initialize
                # 0 = 1 ??
                shared_layers = [weight * param for param in self.models[0].get_shared_layers()]
                tmp_model = copy.deepcopy(self.models[0])
                for state_dict in all_recieved:
                    tmp_model.load_state_dict(state_dict)
                    other_layers = tmp_model.get_shared_layers()
                    for i, layer in enumerate(shared_layers):
                        layer += weight * other_layers[i]

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

            if self.layers_sharing:
                # set the shared layers
                for model in self.models:
                    model.set_shared_layers(shared_layers)

        self._post_step()
        self.communication_round += 1
