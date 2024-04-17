import torch

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.training.Training import Training


class TrainingNIID(Training):
    """
    This class just adds a few function for per sample computaions.
    """

    def compute_per_sample_loss(self, dataset: Dataset, loss_func):
        """
        Compute the per sample loss for the current model (the one that will be shared).

        Args:
            dataset (decentralizepy.datasets.Dataset): The training dataset.
            loss_func: Loss function, must have reduction set to none.

        Returns:
            list: List containing the per sample loss
        """
        self.model.eval()
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        with torch.no_grad():
            per_sample_loss = []
            for data, target in trainset:
                output = self.model(data)
                losses = loss_func(output, target)
                per_sample_loss.extend(losses.tolist())
        return per_sample_loss
