import importlib
import logging
from typing import Dict, List

import torch

from decentralizepy.models.Model import Model
from decentralizepy.sharing.Sharing import Sharing


class IFCASharing(Sharing):
    """
    API defining who to share with and what, and what to do on receiving

    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        models: List[Model],
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
        model : decentralizepy.models.Model
            Model to train
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)

        """
        self.rank = rank
        self.machine_id = machine_id
        self.uid = mapping.get_uid(rank, machine_id)
        self.communication = communication
        self.mapping = mapping
        self.models = models
        self.number_of_models = len(models)
        self.model = None
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

    def get_data_to_send_node(
        self,
        model_idx: int,
    ) -> Dict:
        self._pre_step()
        self.model = self.models[model_idx]
        data = self.serialized_model()
        data["model_idx"] = model_idx
        data["iteration"] = self.communication_round
        return data

    def recieve_data_node(self, data: Dict):
        """Recieve data from the server.
        To be used by the node.

        Args:
           data (dict): data containing the models
        """
        with torch.no_grad():
            self.communication_round = data["iteration"]
            del data["iteration"]
            del data["CHANNEL"]
            incomming_models = self.deserialize_all_models(data)
            for current, incomming in zip(self.models, incomming_models):
                current.load_state_dict(incomming)
            self._post_step()

    def get_data_to_send_server(self):
        """Gets the data that will be sent by the server to the nodes.

        Returns:
            Dict: the data to be sent
        """
        self._pre_step()
        data = self.serialize_all_models()
        data["iteration"] = self.communication_round
        return data

    def _averaging_server(self, peer_deques):
        """Averages the received models on the server.

        Args:
            peer_deques (deque): deque with the received data.
        """
        with torch.no_grad():
            received_models = dict()
            self.model = self.models[0]  # need to instanciate with any model
            for n in peer_deques:
                data = peer_deques[n].popleft()
                iteration, model_idx = (
                    data["iteration"],
                    data["model_idx"],
                )
                del data["iteration"]
                del data["model_idx"]
                del data["CHANNEL"]
                logging.debug("Averaging model from neighbor {} of iteration {}".format(n, iteration))
                data = self.deserialized_model(data)
                if model_idx in received_models:
                    received_models[model_idx].append(data)
                else:
                    received_models[model_idx] = [data]

            # iterate on all the current models of the server
            for idx, model in enumerate(self.models):
                if idx not in received_models:
                    # no model received, keep the previous model
                    continue

                # compute the weight based on the number of incomming models
                weight = 1 / (len(received_models[idx]))

                # add the received models
                total = {}
                for rec_model in received_models[idx]:
                    for key, value in rec_model.items():
                        if key in total:
                            total[key] += value * weight
                        else:
                            total[key] = value * weight
                # assign the new state to the model
                model.load_state_dict(total)

        self._post_step()
        self.communication_round += 1

    def serialize_all_models(self):
        """
        Convert all models to a dictionary. Here we can choose how much to share.

        Returns:
            dict: all models converted to dictionary.
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

    def deserialize_all_models(self, m: dict) -> List[Dict[str, torch.Tensor]]:
        """
        Convert received dict to state_dict.

        Args:
            m (dict): Received dict.

        Returns:
            List[Dict[str, torch.Tensor]]: State_dicts of all models received.

        """
        state_dicts = []
        m = self.decompress_data(m)
        T = m["params"]
        start_index = 0
        for model in self.models:
            state_dict = dict()
            for i, key in enumerate(model.state_dict()):
                end_index = start_index + self.lens[i]
                state_dict[key] = torch.from_numpy(T[start_index:end_index].reshape(self.shapes[i]))
                start_index = end_index
            state_dicts.append(state_dict)
        return state_dicts
