import logging
import torch
import torch.nn.functional as F


class LOSSMIA:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def model_eval(self, model, data_samples):
        with torch.no_grad():
            model = model.to(self.device)
            data, targets = data_samples
            data = data.to(self.device)
            targets = targets.to(self.device)
            output = model(data)
            loss_val = (F.cross_entropy(output, targets, reduction="none").detach().clone())
            return loss_val

    def attack_dataset(
        self,
        victim_model,
        in_dataloader,
        out_dataloader
    ):
        victim_model.eval()

        # Initialize loss values
        loss_vals = {
            "in": torch.zeros((len(in_dataloader.dataset),), dtype=torch.float32, device=self.device),
            "out": torch.zeros((len(out_dataloader.dataset),), dtype=torch.float32, device=self.device),
        }

        with torch.no_grad():
            last = 0
            for data_samples in in_dataloader:
                loss_in = -self.model_eval(victim_model, data_samples)
                loss_vals["in"][last : last + len(data_samples[1])] = loss_in
                # log loss_in
                logging.debug("Member Loss: {}".format(loss_in))
                last += len(data_samples[1])
            loss_vals["in"] = loss_vals["in"][:last].cpu()

            last = 0
            for data_samples in out_dataloader:
                loss_out = -self.model_eval(victim_model, data_samples)
                loss_vals["out"][last : last + len(data_samples[1])] = loss_out
                # log loss_out
                logging.debug("Non-member Loss: {}".format(loss_out))
                last += len(data_samples[1])
            loss_vals["out"] = loss_vals["out"][:last].cpu()
            return loss_vals