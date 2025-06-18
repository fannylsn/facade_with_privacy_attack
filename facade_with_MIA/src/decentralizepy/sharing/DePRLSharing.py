import logging

import torch

from decentralizepy.sharing.Sharing import Sharing


class DePRLSharing(Sharing):
    """
    API defining who to share with and what, and what to do on receiving for DePRL

    """

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model, only update the body of the model

        """
        with torch.no_grad():
            total = dict()
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
                data = self.deserialized_model(data)
                # Metro-Hastings
                weight = 1 / (max(len(peer_deques), degree) + 1)
                weight_total += weight
                for key, value in data.items():
                    if key in total:
                        total[key] += value * weight
                    else:
                        total[key] = value * weight

            for key, value in self.model.state_dict().items():
                total[key] += (1 - weight_total) * value  # Metro-Hastings

            old_state = self.model.state_dict()  # need of deepcopy ?
            # only keep the body of agregation, head stays local
            for key in old_state.keys():
                if self.model.key_in_head(key):
                    total[key] = old_state[key]
            self.model.load_state_dict(total)

        self._post_step()
        self.communication_round += 1
