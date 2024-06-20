# Implemented based on https://github.com/biomedia-mira/stochastic_segmentation_networks

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Dice


class SSNLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 20):
        """
        Args:
            num_mc_samples: Number of Monte Carlo samples
        """
        super().__init__()
        self.num_mc_samples = num_mc_samples
        self.dice_metric = Dice(num_classes=5, average='macro', mdmc_average='samplewise')

    def fixed_reparametrisation_trick(self, distribution, num_samples):
        """
        Args:
            distribution: Distribution object from torch.distributions
            num_samples: Number of samples to draw

        Returns:
            samples: Samples of shape (num_samples, B, num_classes, H, W)
        """
        assert num_samples % 2 == 0, "Number of samples should be even"
        samples = distribution.rsample((num_samples // 2,)) # returns shape (num_samples // 2, B, num_classes, H, W)
        mean = distribution.mean.unsqueeze(0) # mean shape = (1, B, num_classes, H, W)

        samples = samples - mean
        samples = torch.cat([samples, -samples], dim=0) + mean # (num_samples, B, num_classes, H, W)
        return samples


    def forward(self, logits, targets, edge_maps, edge_weight, distribution):
        """
        Args:
            logits: Logits of shape (B, num_classes, H, W)
            targets: Target tensor of shape (B, H, W)
            edge_maps: Edge maps of shape (B, H, W)
            edge_weight: Weight to assign to edge maps
            distribution: Distribution object from torch.distributions

        Returns:
            loss: Scalar value
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        assert num_classes >= 2, "not implemented for binary case with implied background"
        # assert num_classes == distribution.event_shape[0], "Number of classes should match the event shape of the distribution"

        samples = self.fixed_reparametrisation_trick(distribution, self.num_mc_samples) # returns (num_mc_samples, B, num_classes, H, W)
        samples = samples.view(self.num_mc_samples * batch_size, num_classes, -1) # (num_mc_samples * B, num_classes, H * W)

        targets = targets.unsqueeze(1) # (B, 1, H, W)
        targets = targets.expand((self.num_mc_samples,) + targets.shape) # (num_mc_samples, B, 1, H, W)
        targets = targets.reshape((self.num_mc_samples * batch_size, -1)) # (num_mc_samples * B, H * W)

        log_prob = -F.cross_entropy(samples, targets, reduction='none') # (num_mc_samples * B, H * W)
        log_prob = log_prob.reshape((self.num_mc_samples, batch_size, -1)) # (num_mc_samples, B, H * W)

        # Weight loss using edge maps
        edge_maps = edge_maps.unsqueeze(0) # (1, B, H, W)
        edge_maps = edge_maps.expand((self.num_mc_samples,) + edge_maps.shape) # (num_mc_samples, B, H, W)
        edge_maps = edge_maps.reshape((self.num_mc_samples, batch_size, -1)) # (num_mc_samples, B, H * W)

        weights = torch.where(edge_maps == 1, edge_weight, torch.tensor(1.0)) # weights = a tensor of edge_weight where edge_maps is 1, and 1 where edge_maps is 0

        log_prob = log_prob * weights # (num_mc_samples, B, H * W)

        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(self.num_mc_samples)) # scalar
        loss = -loglikelihood

        # add macro dice loss

        # dice_loss = 1 - self.dice_metric(F.softmax(samples.view(-1, num_classes, *logits.size()[2:]), dim=1), targets.view(-1, *logits.size()[2:]))
        # total_loss = (loss / 200000.0) + dice_loss  # Add the Dice loss to the MC loss

        return loss

        # log_prob = -F.cross_entropy(samples, targets, reduction='none') # (num_mc_samples * B, H * W)
        # log_prob = log_prob.view((self.num_mc_samples, batch_size, -1)) # (num_mc_samples, B, H * W)
        # loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(self.num_mc_samples)) # scalar
        # loss = -loglikelihood
        # return loss