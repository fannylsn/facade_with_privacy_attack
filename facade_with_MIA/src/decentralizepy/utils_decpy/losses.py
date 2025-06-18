import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairLogLossDemoPar(nn.Module):
    def __init__(self, reduction="mean", lambda_=0.1, momentum_fair_metrics=0.9):
        super().__init__()
        self.lambda_ = lambda_
        self.momentum_fair_metrics = momentum_fair_metrics

        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(
            reduction="mean"
        )  # always, we compute mse of classes, no sample

        self.training = False

        self.pos_probs = None
        # self.pos_probs = torch.ones(10).float() * 0.1  # 1/10

    def set_fair_metric(self, pos_probs: torch.Tensor):
        self.pos_probs = pos_probs

    def set_fair_metric_other(self, pos_probs_other: torch.Tensor):
        self.pos_probs_other = pos_probs_other

    def get_fair_metric(self):
        return self.pos_probs.detach().clone()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Custom loss"""
        # C = outputs.size(1)
        N = len(targets)
        if N == 0:
            raise ValueError("The batch size should not be zero.")

        # standard loss
        cross_entropy_loss = self.cross_entropy(outputs, targets)  # scalar

        # carefull, coulb be num. unstable
        y_prob = F.softmax(outputs, dim=1)  # predicted labels 0, ..., 9 for CIFAR10

        # one_hot_targets = F.one_hot(targets, num_classes=C).float()
        # logging.debug(f"targets: {targets}")
        # logging.debug(f"y_prob: {y_prob}")

        # training faireness
        # soft_confusion_matrix = torch.matmul(one_hot_targets.T, y_prob)  # C x C

        # logging.debug(f"soft_confusion_matrix shape: {soft_confusion_matrix.shape}")
        # all following are C-dimentional vectors
        # TP = soft_confusion_matrix.diagonal()
        # FP = soft_confusion_matrix.sum(axis=0) - soft_confusion_matrix.diagonal()
        # FN = soft_confusion_matrix.sum(axis=1) - soft_confusion_matrix.diagonal()
        # TN = N - (TP + FP + FN)

        # pos_probs = (TP + FP) / N
        pos_probs = y_prob.sum(dim=0) / N
        # logging.debug(f"batch recall_probs: {pos_probs}")

        # here update the recall_probs
        # logging.debug(f"old recall_probs: {self.pos_probs}")
        pos_probs = self._running_mean_fair_metric(pos_probs)
        # logging.debug(f"new recall_probs:{self.pos_probs}")

        # Discrepancy from the external positive rates
        demo_par_loss = self.mse_loss(pos_probs, self.pos_probs_other)
        logging.debug(
            f"requ grad: {pos_probs.requires_grad}, fair_loss = {demo_par_loss}"
        )

        # loss = cross_entropy_loss / torch.sqrt(self.lambda_) + torch.sqrt(self.lambda_) * demo_par_loss
        loss = cross_entropy_loss + self.lambda_ * demo_par_loss
        return loss

    def _running_mean_fair_metric(self, pos_probs: torch.Tensor):
        # momentum update of 0.9
        if self.training:
            # update running avg
            if self.pos_probs is None:
                self.pos_probs = pos_probs
            else:
                self.pos_probs = (
                    self.pos_probs.detach() * self.momentum_fair_metrics
                    + pos_probs * (1 - self.momentum_fair_metrics)
                )
            return self.pos_probs
        else:
            return pos_probs  # no update outside of training

    def train(self):
        """Set the module in training mode."""
        self.training = True

    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


class FairLogLossEquOdds(nn.Module):
    def __init__(self, reduction="mean", lambda_=0.1, momentum_fair_metrics=0.9):
        super().__init__()
        self.lambda_ = lambda_
        self.momentum_fair_metrics = momentum_fair_metrics

        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(
            reduction="mean"
        )  # always, we compute mse of classes, no sample

        self.training = False

        self.recall_probs = None

    def set_fair_metric(self, recall_probs: torch.Tensor):
        self.recall_probs = recall_probs

    def get_fair_metric(self):
        return self.recall_probs.detach().clone()

    def set_fair_metric_other(self, recall_probs_other: torch.Tensor):
        self.recall_probs_other = recall_probs_other

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Custom loss"""
        C = outputs.size(1)
        N = len(targets)
        if N == 0:
            raise ValueError("The batch size should not be zero.")

        # standard loss
        cross_entropy_loss = self.cross_entropy(outputs, targets)  # scalar

        y_prob = F.softmax(outputs, dim=1)  # predicted labels 0, ..., 9 for CIFAR10
        one_hot_targets = F.one_hot(targets, num_classes=C).float()
        # logging.debug(f"targets: {targets}")
        # logging.debug(f"y_prob: {y_prob}")

        # training faireness
        soft_confusion_matrix = torch.matmul(one_hot_targets.T, y_prob)  # C x C

        # logging.debug(f"soft_confusion_matrix shape: {soft_confusion_matrix.shape}")
        # all following are C-dimentional vectors
        TP = soft_confusion_matrix.diagonal()
        FP = soft_confusion_matrix.sum(axis=0) - soft_confusion_matrix.diagonal()
        FN = soft_confusion_matrix.sum(axis=1) - TP
        TN = N - (TP + FP + FN)
        # logging.debug(f"TP: {TP}, FN: {FN}")

        # TRUE POSITIVE RATE (RECALL)
        denominator = TP + FN
        safe_denominator = torch.where(
            denominator == 0, torch.ones_like(denominator), denominator
        )  # Avoid division by zero
        recall_probs = TP / safe_denominator
        # logging.debug(f"batch recall_probs: {recall_probs}")

        # FALSE POSITIVE RATE
        denominator = TN + FP
        safe_denominator = torch.where(
            denominator == 0, torch.ones_like(denominator), denominator
        )
        fpr = FP / safe_denominator

        # Both must be | .. -  .. | -> easier to just concat them
        recall_probs = torch.cat([recall_probs, fpr], dim=0)
        # print("rec shape", recall_probs.shape)
        # recall_probs = torch.stack([recall_probs, fpr], dim=1).flatten()
        # recall_probs = torch.stack([recall_probs, fpr], dim=1).flatten()

        # here update the recall_probs
        # logging.debug(f"old recall_probs: {self.recall_probs}")
        recall_probs = self._running_mean_fair_metric(recall_probs)
        # logging.debug(f"new recall_probs:{self.recall_probs}")

        equ_odds_loss = self.mse_loss(recall_probs, self.recall_probs_other)
        logging.debug(
            f"requ grad: {recall_probs.requires_grad}, fair_loss = {equ_odds_loss}"
        )

        # loss = cross_entropy_loss / torch.sqrt(self.lambda_) + torch.sqrt(self.lambda_) * equ_odds_loss
        loss = cross_entropy_loss + self.lambda_ * equ_odds_loss
        return loss

    def _running_mean_fair_metric(self, recall_probs: torch.Tensor):
        # momentum update of 0.9
        if self.training:
            # update running avg
            if self.recall_probs is None:
                self.recall_probs = recall_probs
            else:
                self.recall_probs = (
                    self.recall_probs.detach() * self.momentum_fair_metrics
                    + recall_probs * (1 - self.momentum_fair_metrics)
                )
            return self.recall_probs
        else:
            return recall_probs  # no update outside of training

    def train(self):
        """Set the module in training mode."""
        self.training = True

    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


class FairLogLossAcc(nn.Module):
    def __init__(self, reduction="mean", lambda_=0.1, momentum_fair_metrics=0.9):
        super().__init__()
        self.lambda_ = lambda_
        self.momentum_fair_metrics = momentum_fair_metrics

        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(
            reduction="mean"
        )  # always, we compute mse of classes, no sample

        self.training = False

        self.acc_probs = None

    def set_fair_metric(self, acc_probs: torch.Tensor):
        self.acc_probs = acc_probs

    def get_fair_metric(self):
        return self.acc_probs.detach().clone()

    def set_fair_metric_other(self, acc_probs_other: torch.Tensor):
        self.acc_probs_other = acc_probs_other

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Custom loss"""
        C = outputs.size(1)
        N = len(targets)
        if N == 0:
            raise ValueError("The batch size should not be zero.")

        # standard loss
        cross_entropy_loss = self.cross_entropy(outputs, targets)  # scalar

        y_prob = F.softmax(outputs, dim=1)  # predicted labels 0, ..., 9 for CIFAR10
        one_hot_targets = F.one_hot(targets, num_classes=C).float()
        # logging.debug(f"targets: {targets}")
        # logging.debug(f"y_prob: {y_prob}")

        # training faireness
        soft_confusion_matrix = torch.matmul(one_hot_targets.T, y_prob)  # C x C

        # logging.debug(f"soft_confusion_matrix shape: {soft_confusion_matrix.shape}")
        # all following are C-dimentional vectors
        TP = soft_confusion_matrix.diagonal()
        FP = soft_confusion_matrix.sum(axis=0) - soft_confusion_matrix.diagonal()
        FN = soft_confusion_matrix.sum(axis=1) - TP
        TN = N - (TP + FP + FN)
        # logging.debug(f"TP: {TP}, FN: {FN}")

        acc_probs = (TP + TN) / N
        # logging.debug(f"batch recall_probs: {recall_probs}")

        # here update the recall_probs
        # logging.debug(f"old recall_probs: {self.recall_probs}")
        acc_probs = self._running_mean_fair_metric(acc_probs)
        # logging.debug(f"new recall_probs:{self.recall_probs}")

        acc_diff_loss = self.mse_loss(acc_probs, self.acc_probs_other)
        logging.debug(
            f"requ grad: {acc_probs.requires_grad}, fair_loss = {acc_diff_loss}"
        )

        # loss = cross_entropy_loss / torch.sqrt(self.lambda_) + torch.sqrt(self.lambda_) * acc_diff_loss
        loss = cross_entropy_loss + self.lambda_ * acc_diff_loss
        return loss

    def _running_mean_fair_metric(self, acc_probs: torch.Tensor):
        # momentum update of 0.9
        if self.training:
            # update running avg
            if self.acc_probs is None:
                self.acc_probs = acc_probs
            else:
                self.acc_probs = (
                    self.acc_probs.detach() * self.momentum_fair_metrics
                    + acc_probs * (1 - self.momentum_fair_metrics)
                )
            return self.acc_probs
        else:
            return acc_probs  # no update outside of training

    def train(self):
        """Set the module in training mode."""
        self.training = True

    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


class FairLogLossDiffLoss(nn.Module):
    def __init__(self, reduction="mean", lambda_=0.1, momentum_fair_metrics=0.9):
        super().__init__()
        self.lambda_ = lambda_
        self.momentum_fair_metrics = momentum_fair_metrics

        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(
            reduction="mean"
        )  # always, we compute mse of cluster, no sample

        self.training = False

        self.mean_clust_loss = None

    def set_fair_metric(self, mean_clust_loss: torch.Tensor):
        self.mean_clust_loss = mean_clust_loss.mean()

    def get_fair_metric(self):
        # print(self.mean_clust_loss)
        return self.mean_clust_loss.detach().clone()

    def set_fair_metric_other(self, mean_cluster_loss_other: torch.Tensor):
        self.mean_clust_loss_other = (
            mean_cluster_loss_other.mean()
        )  # first iter is init tensor with 10 entries

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Custom loss"""
        # C = outputs.size(1)
        N = len(targets)
        if N == 0:
            raise ValueError("The batch size should not be zero.")

        # standard loss
        cross_entropy_loss = self.cross_entropy(outputs, targets)  # scalar

        # y_prob = F.softmax(outputs, dim=1)  # predicted labels 0, ..., 9 for CIFAR10
        # one_hot_targets = F.one_hot(targets, num_classes=C).float()
        # logging.debug(f"targets: {targets}")
        # logging.debug(f"y_prob: {y_prob}")

        # training faireness
        # soft_confusion_matrix = torch.matmul(one_hot_targets.T, y_prob)  # C x C

        # logging.debug(f"soft_confusion_matrix shape: {soft_confusion_matrix.shape}")
        # all following are C-dimentional vectors
        # TP = soft_confusion_matrix.diagonal()
        # FP = soft_confusion_matrix.sum(axis=0) - soft_confusion_matrix.diagonal()
        # FN = soft_confusion_matrix.sum(axis=1) - TP
        # TN = N - (TP + FP + FN)
        # logging.debug(f"TP: {TP}, FN: {FN}")

        # denominator = TP + FN
        # safe_denominator = torch.where(
        #     denominator == 0, torch.ones_like(denominator), denominator
        # )  # Avoid division by zero
        # recall_probs = TP / safe_denominator
        # logging.debug(f"batch recall_probs: {recall_probs}")

        # here update the recall_probs
        # logging.debug(f"old recall_probs: {self.recall_probs}")
        mean_clust_loss = self._running_mean_fair_metric(cross_entropy_loss)
        # logging.debug(f"new recall_probs:{self.recall_probs}")
        # print(f"scalar:{mean_clust_loss}")
        # print(f"other {self.mean_clust_loss_other}")
        clust_loss_diff = self.mse_loss(mean_clust_loss, self.mean_clust_loss_other)
        logging.debug(
            f"requ grad: {clust_loss_diff.requires_grad}, fair_loss = {clust_loss_diff}"
        )

        # loss = cross_entropy_loss / torch.sqrt(self.lambda_) + torch.sqrt(self.lambda_) * clust_loss_diff
        loss = cross_entropy_loss + self.lambda_ * clust_loss_diff
        return loss

    def _running_mean_fair_metric(self, nllloss: torch.Tensor):
        # momentum update of 0.9
        mean_nllloss = nllloss.mean()
        if self.training:
            # update running avg
            if self.mean_clust_loss is None:
                self.mean_clust_loss = mean_nllloss
            else:
                self.mean_clust_loss = (
                    self.mean_clust_loss.detach() * self.momentum_fair_metrics
                    + mean_nllloss * (1 - self.momentum_fair_metrics)
                )
            return self.mean_clust_loss
        else:
            return mean_nllloss  # no update outside of training

    def train(self):
        """Set the module in training mode."""
        self.training = True

    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


class FairLogLossModelDiff(nn.Module):
    def __init__(self, reduction="mean", lambda_=0.1):
        super().__init__()
        self.lambda_ = torch.tensor([lambda_])
        # self.lambda_ = nn.Parameter(torch.tensor(0.1))
        logging.debug(f"lambda_: {self.lambda_} global for all layers")

        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

        self.training = False

    def set_model(self, model: torch.nn.Module):
        self.model = model  # same object

    def compute_other_model(self, all_other_models: List):
        model_other = all_other_models[0].deepcopy()
        # mean of state dict of all other models
        state_dict = model_other.state_dict()
        for model in all_other_models[1:]:
            state_dict_other = model.state_dict()
            for key in state_dict:
                state_dict[key] += state_dict_other[key]
        for key in state_dict:
            state_dict[key] /= len(all_other_models)
        model_other.load_state_dict(state_dict)
        return model_other

    def set_model_other(self, model_other: torch.nn.Module):
        self.model_other = model_other  # deep copy is done outside
        for params in self.model_other.parameters():
            params.requires_grad = False

    def set_fair_metric(self, mean_clust_loss: torch.Tensor):
        """kept for compatibility"""
        pass

    def get_fair_metric(self):
        """kept for compatibility"""
        return torch.tensor([0.0])

    def set_fair_metric_other(self, mean_cluster_loss_other: torch.Tensor):
        """kept for compatibility"""
        pass

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Custom loss"""
        # C = outputs.size(1)
        N = len(targets)
        if N == 0:
            raise ValueError("The batch size should not be zero.")

        # standard loss
        self.cross_entropy_loss = self.cross_entropy(outputs, targets)  # scalar

        param_diff = []
        for param1, param2 in zip(
            self.model.parameters(), self.model_other.parameters()
        ):
            param_diff.append(torch.norm(torch.sub(param1, param2), p=2))  # old

        stacked_diffs = torch.stack(param_diff)
        self.param_diff_mean = torch.mean(stacked_diffs, dim=0)

        if self.training:
            logging.debug(
                f"cel loss: {self.cross_entropy_loss}, param_diff loss: {self.param_diff_mean}"
            )
        # if self.training:
        #     self.cross_entropy_loss.retain_grad()
        #     self.param_diff_mean.retain_grad()
        # loss = self.cross_entropy_loss / torch.sqrt(self.lambda_) + torch.sqrt(self.lambda_) * self.param_diff_mean
        loss = self.cross_entropy_loss + self.lambda_ * self.param_diff_mean
        return loss

    def train(self):
        """Set the module in training mode."""
        self.training = True

    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


class FairLogLossLayerDiff(nn.Module):
    def __init__(self, reduction="mean", num_blocks=1, lambda_=0.1):
        super().__init__()
        self.num_blocks = num_blocks
        self.lambda_ini = lambda_
        self.lambda_ = torch.tensor([lambda_] * self.num_blocks)
        self.lambda_[-1] = 0.0  # TEST
        # self.lambda_ = nn.Parameter(torch.tensor(0.1))
        logging.debug(f"lambda_: {self.lambda_} for each layer")

        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

        self.training = False

    def scale_lambda(self, scaler: torch.Tensor):
        lambda_ = self.lambda_ini * scaler
        self.lambda_ = torch.tensor([lambda_] * self.num_blocks)
        self.lambda_[-1] = 0.0  # TEST

    def set_model(self, model: torch.nn.Module):
        self.model = model  # same object

    def compute_other_model(self, all_other_models: List):
        model_other = all_other_models[0].deepcopy()
        # mean of state dict of all other models
        state_dict = model_other.state_dict()
        for model in all_other_models[1:]:
            state_dict_other = model.state_dict()
            for key in state_dict:
                state_dict[key] += state_dict_other[key]
        for key in state_dict:
            state_dict[key] /= len(all_other_models)
        model_other.load_state_dict(state_dict)
        return model_other

    def set_model_other(self, model_other: torch.nn.Module):
        self.model_other = model_other  # deep copy is done outside
        for params in self.model_other.parameters():
            params.requires_grad = False

    def set_fair_metric(self, mean_clust_loss: torch.Tensor):
        """kept for compatibility"""
        pass

    def get_fair_metric(self):
        """kept for compatibility"""
        return torch.tensor([0.0])

    def set_fair_metric_other(self, mean_cluster_loss_other: torch.Tensor):
        """kept for compatibility"""
        pass

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Custom loss"""
        # C = outputs.size(1)
        N = len(targets)
        if N == 0:
            raise ValueError("The batch size should not be zero.")

        # standard loss
        self.cross_entropy_loss = self.cross_entropy(outputs, targets)  # scalar

        scaled_param_diff = []
        for lamb, param1, param2 in zip(
            self.lambda_, self.model.get_layers(), self.model_other.get_layers()
        ):
            scaled_param_diff.append(
                lamb
                * torch.sqrt(
                    torch.sum(
                        torch.pow(torch.sub(param1.weight, param2.weight), exponent=2)
                    )
                    + torch.sum(
                        torch.pow(torch.sub(param1.bias, param2.bias), exponent=2)
                    )
                )
            )

        scaled_stacked_diffs = torch.stack(scaled_param_diff)
        self.scaled_param_diff_mean = torch.mean(scaled_stacked_diffs, dim=0)

        if self.training:
            logging.debug(
                f"cel loss: {self.cross_entropy_loss}, param_diff loss: {self.scaled_param_diff_mean}"
            )
        # if self.training:
        #     self.cross_entropy_loss.retain_grad()
        #     self.param_diff_mean.retain_grad()
        # loss = self.cross_entropy_loss / torch.sqrt(self.lambda_) + torch.sqrt(self.lambda_) * self.param_diff_mean
        loss = self.cross_entropy_loss + self.scaled_param_diff_mean
        return loss

    def train(self):
        """Set the module in training mode."""
        self.training = True

    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


class FeatureAlignmentLoss(nn.Module):
    def __init__(self, reduction="mean", lambda_=1.0):
        super().__init__()
        self.lambda_ = torch.tensor([lambda_])
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

        self.training = False

        self.inner_count = 1

    def set_model(self, model: torch.nn.Module):
        self.model = model  # same object

    def compute_other_model(self, all_other_models: List):
        model_other = all_other_models[0].deepcopy()
        # mean of state dict of all other models
        state_dict = model_other.state_dict()
        for model in all_other_models[1:]:
            state_dict_other = model.state_dict()
            for key in state_dict:
                state_dict[key] += state_dict_other[key]
        for key in state_dict:
            state_dict[key] /= len(all_other_models)
        model_other.load_state_dict(state_dict)
        return model_other

    def set_model_other(self, model_other: torch.nn.Module):
        self.model_other = model_other  # deep copy is done outside
        for params in self.model_other.parameters():
            params.requires_grad = False

    def set_fair_metric(self, mean_clust_loss: torch.Tensor):
        """kept for compatibility"""
        pass

    def get_fair_metric(self):
        """kept for compatibility"""
        return torch.tensor([0.0])

    def set_fair_metric_other(self, mean_cluster_loss_other: torch.Tensor):
        """kept for compatibility"""
        pass

    def forward(self, outputs, targets):
        # Calculate cross-entropy loss for both models
        cross_entropy_loss = self.cross_entropy(outputs, targets)

        # Get the features computed in the previous forward pass
        features_self = self.model.get_features()
        features_other = self.model_other.get_features()
        assert features_self is not None, "Features not extracted from model A"
        assert features_other is not None, "Features not extracted from model B"

        # Calculate the mean squared error between the features from both models
        feature_alignment_loss = F.mse_loss(features_self, features_other)

        # reset to make sure to compute the news ones
        # self.model.reset_features()
        # self.model_other.reset_features()

        # Weight the feature alignment loss
        weighted_feature_alignment_loss = self.lambda_ * feature_alignment_loss

        if True:
            total_loss = cross_entropy_loss + weighted_feature_alignment_loss
            return total_loss

        if self.training:
            if self.inner_count % 5 == 0:
                # Combine the cross-entropy loss and the weighted feature alignment loss
                total_loss = weighted_feature_alignment_loss
                self.inner_count = 1
                logging.debug(
                    f"Feature alignment loss: {weighted_feature_alignment_loss}"
                )
            else:
                self.inner_count += 1
                total_loss = cross_entropy_loss
                logging.debug(f"Cross-entropy loss: {cross_entropy_loss}")
        else:
            total_loss = cross_entropy_loss

        return total_loss

    def train(self):
        """Set the module in training mode."""
        self.training = True

    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False
