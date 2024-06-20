# Implemented based on https://github.com/biomedia-mira/stochastic_segmentation_networks

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F

from .unet_model import UNet
from utils.reshaped_dist import ReshapedDistribution


class StochasticUNet(UNet):
    def __init__(self, in_channels, num_classes, rank=10, epsilon=1e-5, diagonal=False, dropout_prob=0.2):
        """
        Args:
            in_channels: no. of input channels
            num_classes: no. of output classes
            rank: rank of the low-rank covariance matrix
            epsilon: small value added to the diagonal of the cov matrix for numerical stability
            diagonal: whether to use only the diagonal (independent normals)
            dropout_prob: dropout probability    
        """
        super().__init__(in_channels, num_classes=32, dropout_prob=dropout_prob) # Get 32 channel output from base UNet to use in stochastic layer

        self.num_classes = num_classes
        self.rank = rank
        self.epsilon = epsilon
        self.diagonal = diagonal
        
        self.mean_l = nn.Conv2d(32, num_classes, kernel_size=1)
        self.log_cov_diag_l = nn.Conv2d(32, num_classes, kernel_size=1)
        self.cov_factor_l = nn.Conv2d(32, num_classes * rank, kernel_size=1)

    def forward(self, image, **kwargs):
        """
        Args:
            image: Image tensor of shape (batch_size, in_channels, height, width)

        Returns:
            logit_mean: Mean of the output distribution
            output_dict: Dict containing the distribution, mean, low-rank cov matrix and diagonal cov matrix
        """

        logits = F.relu(super().forward(image, **kwargs)) # Logits of shape (B, 32, H, W)
        batch_size = logits.size(0)
        event_shape = (self.num_classes,) + logits.shape[2:] # (num_classes, H, W)

        mean = self.mean_l(logits) # (B, num_classes, H, W)
        mean = mean.view((batch_size, -1)) # (B, num_classes * H * W)

        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon # (B, num_classes, H, W)
        cov_diag = cov_diag.view((batch_size, -1)) # (B, num_classes * H * W)

        cov_factor = self.cov_factor_l(logits) # (B, num_classes * rank, H, W)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1)) # (B, rank, num_classes, H * W)
        cov_factor = cov_factor.flatten(2, 3) # (B, rank, num_classes * H * W)
        cov_factor = cov_factor.transpose(1, 2) # (B, num_classes * H * W, rank)

        try:
            base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
        except Exception as e:
            print(e)
            print('Covariance became not invertible, using independent normals for this batch!')
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape # (B, num_classes, H, W)
        logit_mean = mean.view(shape).detach() # (B, num_classes, H, W)
        cov_diag_view = cov_diag.view(shape).detach() # (B, num_classes, H, W)
        # (B, num_classes * H * W, rank) -> after transpose (B, rank, num_classes * H * W) -> after view (B, num_classes * rank, H, W)
        cov_factor_view = cov_factor.transpose(2, 1).view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
    
        output_dict = {'logit_mean': logit_mean,
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict
