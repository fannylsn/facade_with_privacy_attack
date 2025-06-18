import logging
import copy
import os
from typing import List

import torch
import numpy as np

from decentralizepy.models.Model import Model
from decentralizepy.sharing.CurrentModelSharing import CurrentModelSharing
from decentralizepy.attacks.LossMIA import LOSSMIA

class SharingAttackRandomLoss(CurrentModelSharing):
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
        attack_after=32,
        perform_attack=True, 
        non_member_attack_data="" 
    ):
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
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.attack_after = attack_after
        self.perform_attack = perform_attack
        
        torch.manual_seed(self.dataset.random_seed)
        np.random.seed(self.dataset.random_seed)

        self.client_trainset_dataloaders = {
            client:self.dataset.get_trainset(
                batch_size=self.dataset.test_batch_size,
                shuffle=False,
                node_id=client
            ) for client in range(self.mapping.get_n_procs())
        }

        self.attack_results = {"loss_vals": {}}

        cluster_testset_dataloaders = {
            cluster_id: self.dataset.get_testset(
                cluster_id=cluster_id,
                attack=self.perform_attack
            ) for cluster_id in range(self.dataset.number_of_clusters)
        }
        attack_testset = torch.utils.data.ConcatDataset(list(cluster_testset_dataloaders.values()))

        if non_member_attack_data == "UNION":
            self.attack_non_member_dataloader = torch.utils.data.DataLoader(
                attack_testset,
                batch_size=self.dataset.test_batch_size,
                shuffle=False
            )
        elif non_member_attack_data == "TEST_DATA_CLUSTER_0":
            self.attack_non_member_dataloader = cluster_testset_dataloaders[0]
        elif non_member_attack_data == "TEST_DATA_CLUSTER_1":
            self.attack_non_member_dataloader = cluster_testset_dataloaders[1]
        else:
            raise ValueError("Invalid non_member_attack_data option. Choose from 'UNION', 'TEST_DATA_CLUSTER_0', or 'TEST_DATA_CLUSTER_1'.")

        self.train_dir = self.dataset.train_dir

        self.mia = LOSSMIA()
        self.victim_model = None
        self.seed = self.dataset.random_seed

            
    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model
        """
        with torch.no_grad():
            received_models = dict()
            clients = dict()
            
            if self.victim_model is None:
                self.victim_model = copy.deepcopy(self.models[1])

            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                
                _, iteration, model_idx = (data["degree"],data["iteration"],data["model_idx"])
                del data["degree"]
                del data["iteration"]
                del data["model_idx"]
                del data["CHANNEL"]

                logging.debug("Averaging model from neighbor {} of iteration {}".format(n, iteration))
                data = self.deserialized_model(data)
                
                if model_idx in received_models:
                    received_models[model_idx].append(data)
                    clients[model_idx].append(n)
                else:
                    received_models[model_idx] = [data]
                    clients[model_idx] = [n]

            nb_clients = len([client for value in clients.values() for client in value])
            nb_received_models = len([model for value in received_models.values() for model in value])

            assert nb_clients == nb_received_models, "Number of clients and number of received models do not match"
              
            if self.communication_round % self.attack_after == 0 and self.perform_attack:

                for model_idx, models_list in received_models.items():
                    for i, received_model in enumerate(models_list):

                        client_id = clients[model_idx][i]
                        cluster_idx = self.dataset.clusters_idx[client_id]

                        if client_id in self.dataset.victim_nodes[cluster_idx]:
                            logging.info(f"Client {self.uid} performing MIA on client {client_id}")                   
                            self.victim_model.load_state_dict(received_model) 
                            attack_loss = self.mia.attack_dataset(
                                self.victim_model,  # get the model of the client we attack
                                self.client_trainset_dataloaders[client_id],  # get trainset of the client we attack
                                self.attack_non_member_dataloader  # get non-member dataset
                            )
                            
                            if client_id not in self.attack_results["loss_vals"]:
                                self.attack_results["loss_vals"][client_id] = dict()
                            
                            if self.communication_round not in self.attack_results["loss_vals"][client_id]:
                                self.attack_results["loss_vals"][client_id][self.communication_round] = []

                            self.attack_results["loss_vals"][client_id][self.communication_round].append(attack_loss)
                            torch.save(self.attack_results, os.path.join(self.log_dir, "{}_attacker.pth".format(self.uid)))

            if self.layers_sharing:
                # average of the common layers
                all_recieved = []
                for models in received_models.values():
                    all_recieved.extend(models)
                weight = 1 / (len(all_recieved) + 1)
                
                shared_layers = [weight * param.clone() for param in self.models[0].get_shared_layers()]
                tmp_model = self.models[0].deepcopy()
                for state_dict in all_recieved:
                    tmp_model.load_state_dict(state_dict)
                    other_layers = tmp_model.get_shared_layers()
                    for i, layer in enumerate(shared_layers):
                        layer += weight * other_layers[i].clone()

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
                    total[key] = value.clone() * weight
                # add the received models
                if idx in received_models:
                    for rec_model in received_models[idx]:
                        for key, value in rec_model.items():
                            total[key] += value.clone() * weight
                # assign the new state to the model
                model.load_state_dict(total)

            if self.layers_sharing:
                # set the shared layers
                for model in self.models:
                    model.set_shared_layers(shared_layers)

        self._post_step()
        self.communication_round += 1