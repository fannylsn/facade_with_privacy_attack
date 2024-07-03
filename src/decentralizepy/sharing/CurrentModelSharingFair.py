import copy
import logging
from typing import Dict, List

import torch

from decentralizepy.models.Model import Model
from decentralizepy.sharing.CurrentModelSharing import CurrentModelSharing


class CurrentModelSharingFair(CurrentModelSharing):
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
        fair_metric_dict: Dict[int, torch.Tensor],
        fair_metric_dict_other: Dict[int, torch.Tensor],
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
        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            models,
            dataset,
            log_dir,
            layers_sharing,
            compress,
            compression_package,
            compression_class,
            float_precision,
        )
        self.fair_metric_dict = fair_metric_dict
        self.fair_metric_dict_other = fair_metric_dict_other

    def get_data_to_send(
        self,
        model_idx: int,
        fair_metric: torch.Tensor,
        degree=None,
    ):
        self._pre_step()
        self.model = self.models[model_idx]
        data = self.serialized_model()
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        data["model_idx"] = model_idx
        data["fair_metric"] = fair_metric
        data["degree"] = degree if degree is not None else len(self.graph.neighbors(my_uid))
        data["iteration"] = self.communication_round
        return data

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            received_models = dict()
            received_fair_metrics = dict()
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                _, iteration, model_idx, fair_metric = (
                    data["degree"],
                    data["iteration"],
                    data["model_idx"],
                    data["fair_metric"],
                )
                del data["degree"]
                del data["iteration"]
                del data["model_idx"]
                del data["CHANNEL"]
                del data["fair_metric"]
                logging.debug("Averaging model from neighbor {} of iteration {}".format(n, iteration))
                data = self.deserialized_model(data)
                if model_idx in received_models:
                    received_models[model_idx].append(data)
                else:
                    received_models[model_idx] = [data]

                # fair metrics
                if model_idx in received_fair_metrics:
                    received_fair_metrics[model_idx].append(fair_metric)
                else:
                    received_fair_metrics[model_idx] = [fair_metric]

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

            # fairness metric
            for idx, fair_metric in received_fair_metrics.items():
                logging.debug(f"Fairness list for model {idx}: {fair_metric}")
                self.fair_metric_dict[idx] = torch.stack(fair_metric, dim=0).mean(dim=0)
                logging.debug(f"Fairness metric for model {idx}: {self.fair_metric_dict[idx]}")

            # update other, the mean of all the other
            for idx in self.fair_metric_dict:
                other_fair_metric = torch.stack(
                    [self.fair_metric_dict[i] for i in self.fair_metric_dict if i != idx], dim=0
                ).mean(dim=0)
                self.fair_metric_dict_other[idx] = other_fair_metric

        self._post_step()
        self.communication_round += 1
