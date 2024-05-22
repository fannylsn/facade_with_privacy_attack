import importlib

from decentralizepy import utils
from decentralizepy.node.DPSGDWithPeerSamplerNIID import DPSGDWithPeerSamplerNIID
from decentralizepy.training.TrainingDePRL import TrainingDePRL  # noqa: F401


class DPSGDNodeDePRL(DPSGDWithPeerSamplerNIID):
    """
    This class defines the node for DPSGD DISPFL with peer sampler for non iid datasets.
    It just redifines the run method to log the cluster assigned to the node and some other methods to log metrics.

    """

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
        )  # type: TrainingDePRL
