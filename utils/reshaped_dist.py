from typing import Tuple
import torch
import torch.distributions as td


class ReshapedDistribution(td.Distribution):
    def __init__(self, base_distribution: td.Distribution, new_event_shape: Tuple[int, ...]):

        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape

        self.loc = base_distribution.loc
        self.cov_diag = base_distribution.cov_diag
        self.cov_factor = base_distribution.cov_factor

        super().__init__(batch_shape=base_distribution.batch_shape, event_shape=new_event_shape)

    @property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return self.base_distribution.arg_constraints

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        samples = self.base_distribution.rsample(sample_shape)
        return samples.view(sample_shape + self.new_shape)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ReshapedDistribution, _instance)
        new.base_distribution = self.base_distribution.expand(batch_shape)
        new.new_shape = batch_shape + self.new_shape
        return new
    
    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()