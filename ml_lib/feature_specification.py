from typing import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .misc import all_equal

@dataclass
class FeatureType():
    name: str
    dim: int
    extract: Callable[..., torch.Tensor] = lambda x: x
    loss_coef: float = 1.
    """Extracts the feature from the input, returns a tensor of shape (batch, dim)"""""

    def compute_loss(self, input, target):
        """Always takes an unnormalized logit for input, 
        (ie the direct output of a MLP for ex, not the softmax or the log softmax)"""
        del input, target
        raise NotImplementedError

    def decode(self, input):
        return input

    def encode(self, input):
        return input

    def get_features(self, input):
        return self.encode(self.extract(input))

class MSEFeature(FeatureType):
    def compute_loss(self, input, target):
        return (input - target).square().mean()

class OneHotFeature(FeatureType):
    def compute_loss(self, input, target):
        return F.cross_entropy(input=input, target=target).mean()

    def decode(self, input):
        return input.argmax(dim=-1)
    def encode(self, input):
        return F.one_hot(input, num_classes=self.dim)

class BinaryFeature(FeatureType):
    def compute_loss(self, input, target):
        return F.binary_cross_entropy_with_logits(input=input, target=target)

    def decode(self, input):
        return input > 0

@dataclass
class FeatureSpecification():
    features: list[FeatureType]

    @property
    def dim(self):
        return sum(feature.dim for feature in self.features)

    def compute_loss(self, input, target):
        *batch_input, n_features = input.shape
        *batch_target, n_features_ = target.shape
        assert all_equal(n_features, n_features_, self.dim), f"Invalid shapes {input.shape} {target.shape}, needed {self.dim} features"
        # assert broadcastable(batch_input, batch_target), f"batch shapes {batch_input} {batch_target} are not broadcastable"
        # broadcastable is in ml_lib.misc but I havent updated it yet
        dims = [feature.dim for feature in self.features]

        inputs = input.split(dims, dim=-1)
        targets = target.split(dims, dim=-1)

        return sum(feature.loss_coef * feature.compute_loss(input, target) 
                   for feature, input, target in zip(self.features, inputs, targets))

    def cut_up(self, input, decode=False):
        *batch, n_features = input.shape
        assert self.dim == n_features, f"Invalid shape {input.shape}, needed {self.dim} features"
        dims = [feature.dim for feature in self.features]
        inputs = input.split(dims, dim=-1)
        if decode: return {feature.name: feature.decode(input) for feature, input in zip(self.features, inputs)}
        return {feature.name: input for feature, input in zip(self.features, inputs)}

    def get_features(self, x):
        return torch.cat([feature.get_features(x) for feature in self.features], dim=-1)

    def stack(self, inputs):
        return torch.cat([inputs[feature.name] for feature in self.features], dim=-1)
