import importlib
import logging
from typing import List

import torch

from decentralizepy.sharing.Sharing import Sharing


class MultiModelSharing(Sharing):
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
        models: List[torch.nn.Module],
        dataset,
        log_dir,
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
            Dataset for sharing data. Not implemented yet!
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
        self.dataset = dataset
        self.communication_round = 0
        self.log_dir = log_dir

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

    def serialize_all_models(self):
        """
        Convert all models to a dictionary. Here we can choose how much to share.

        Returns
        -------
        dict
            Model converted to dict

        """
        to_cat = []
        with torch.no_grad():
            for model in self.models:
                for _, v in model.state_dict().items():
                    t = v.flatten()
                    to_cat.append(t)
        flat = torch.cat(to_cat)
        data = dict()
        data["params"] = flat.numpy()
        logging.debug("Model sending this round: {}".format(data["params"]))
        return self.compress_data(data)

    def deserialize_all_models(self, m):
        """
        Convert received dict to state_dict.

        Parameters
        ----------
        m : dict
            received dict

        Returns
        -------
        state_dict
            state_dict of received

        """
        state_dicts = []
        m = self.decompress_data(m)
        T = m["params"]
        start_index = 0
        for model in self.models:
            state_dict = dict()
            for i, key in enumerate(model.state_dict()):
                end_index = start_index + self.lens[i]
                state_dict[key] = torch.from_numpy(
                    T[start_index:end_index].reshape(self.shapes[i])
                )
                start_index = end_index
            state_dicts.append(state_dict)
        return state_dicts

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            totals = [dict() for _ in range(self.number_of_models)]
            weight_total = 0
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                degree, iteration = data["degree"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialize_all_models(data)
                # Metro-Hastings
                weight = 1 / (max(len(peer_deques), degree) + 1)
                weight_total += weight
                for n, model in enumerate(data):
                    for key, value in model.items():
                        if key in totals[n]:
                            totals[n][key] += value * weight
                        else:
                            totals[n][key] = value * weight
            for n, model in enumerate(self.models):
                for key, value in model.state_dict().items():
                    totals[n][key] += (1 - weight_total) * value  # Metro-Hastings

        for n, model in enumerate(self.models):
            model.load_state_dict(totals[n])
        self._post_step()
        self.communication_round += 1

    def get_data_to_send(self, degree=None):
        self._pre_step()
        data = self.serialize_all_models()
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        data["degree"] = degree if degree is None else len(self.graph.neighbors(my_uid))
        data["iteration"] = self.communication_round
        return data
