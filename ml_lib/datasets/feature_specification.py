from typing import Callable, ClassVar, Any, Self
from dataclasses import dataclass, field as dataclass_field

import torch
from torch import Tensor
import torch.nn.functional as F

from ml_lib.misc import all_equal
from ml_lib.misc.torch_functions import broadcastable
from ml_lib.register import Register, LoadableMixin, Loader


@dataclass
class FeatureType():
    name: str
    dim: int
    extract: Callable[..., torch.Tensor]|None = dataclass_field(default=None, repr=False)
    loss_coef: float = 1.
    """Extracts the feature from the input, returns a tensor of shape (batch, dim)"""""

    def _extract(self, x):
        from ml_lib.datasets.datapoint import Datapoint
        if self.extract is not None:
            return self.extract(x)
        if isinstance(x, Datapoint):
            return x.get_feature(self.name)
        if isinstance(x, dict):
            return x[self.name]
        if hasattr(x, self.name):
            return getattr(x, self.name)
        raise ValueError(f"Couldn't find how to get data {self.name} out of object {x}")

    def compute_loss(self, input, target, reduce = True) -> Tensor:
        """Always takes an unnormalized logit for input, 
        (ie the direct output of a MLP for ex, not the softmax or the log softmax)
        """
        del input, target, reduce
        raise NotImplementedError


    def decode(self, input):
        return input

    def encode(self, input):
        return input

    def get_features(self, input):
        return self.encode(self._extract(input))

    def __eq__(self, other):
        # don't check the "extract" function
        if not self.__class__ == other.__class__:
            return False
        return self.name == other.name \
                and self.dim == other.dim \
                and self.loss_coef == other.loss_coef

    def __getstate__(self):
        return dict(
                name=self.name, 
                dim=self.dim,
                loss_coef=self.loss_coef
            )
    def __setstate__(self, state):
        self.name = state["name"]
        self.dim = state["dim"]
        self.loss_coef = state["loss_coef"]

    def to_dict(self):
        return dict(
            type=self.__class__.__name__, 
            **self.__getstate__()
                )

feature_type_register = Register(FeatureType)


@feature_type_register
class MSEFeature(FeatureType):
    def compute_loss(self, input, target, reduce=True) -> Tensor:
        loss = (input - target).square()
        if reduce: loss = loss.mean()
        else: loss = loss.mean(dim=-1)
        return loss

@feature_type_register
class OneHotFeature(FeatureType):
    def compute_loss(self, input, target, reduce=True) -> Tensor:
        return F.cross_entropy(input=input, target=target, 
                               reduction="mean" if reduce else "none")

    def decode(self, input):
        return input.argmax(dim=-1)
    def encode(self, input):
        return F.one_hot(input, num_classes=self.dim)

@feature_type_register
class BinaryFeature(FeatureType):
    def compute_loss(self, input, target, reduce=True) -> Tensor:
        return F.binary_cross_entropy_with_logits(input=input, target=target,
                                reduction = "mean" if reduce else "none")

    def decode(self, input):
        return input > 0

@dataclass
class FeatureSpecification(LoadableMixin):
    """
    Used to keep track of the signification of the features used in a datapoint 
    (node features / edge features / set features / simple feature vector).
    It keeps track of how the feature is encoded

    for example if the datapoint has 2 continuous coordinates, a discrete label with 5 possible values, 
    and a flag that may be true or false, we can encode it with the following specification
    
    .. code-block:: python

        FeatureSpecification([MSEFeature("coordinates", 2), OneHotFeature("label", 5), BinaryFeature("flag", 1)])

    (the names given as ``string`` can be chosen depending on what the feature represents)
    """ 
    feature_type_register: ClassVar = feature_type_register
    features: list[FeatureType]

    @property
    def dim(self):
        return sum(feature.dim for feature in self.features)

    def compute_loss(self, input, target, reduce=True) -> Tensor:
        *batch_input, n_features = input.shape
        *batch_target, n_features_ = target.shape
        assert all_equal(n_features, n_features_, self.dim), f"Invalid shapes {input.shape} {target.shape}, needed {self.dim} features"
        assert broadcastable(batch_input, batch_target), f"batch shapes {batch_input} {batch_target} are not broadcastable"
        dims = [feature.dim for feature in self.features]

        inputs = input.split(dims, dim=-1)
        targets = target.split(dims, dim=-1)

        return sum(
                (feature.loss_coef * 
                feature.compute_loss(input, target, reduce=reduce) 
                for feature, input, target in zip(self.features, inputs, targets)),
                start=torch.tensor(0., device=input.device, dtype=input.dtype)
                )

    def cut_up(self, input, decode=False):
        *batch, n_features = input.shape
        assert self.dim == n_features, f"Invalid shape {input.shape}, needed {self.dim} features"
        dims = [feature.dim for feature in self.features]
        inputs = input.split(dims, dim=-1)
        if decode: return {feature.name: feature.decode(input) for feature, input in zip(self.features, inputs)}
        return {feature.name: input for feature, input in zip(self.features, inputs)}

    def decode(self, input):
        return self.cut_up(input, decode=True)

    def get_features(self, x):
        return torch.cat([feature.get_features(x) for feature in self.features], dim=-1)

    def stack(self, inputs):
        return torch.cat([inputs[feature.name] for feature in self.features], dim=-1)

    def __repr__(self):
        features_repr = [repr(feature) for feature in self.features]
        return f"FeatureSpecification({', '.join(features_repr)})"

    def to_config(self):
        return dict(
            type=self.__class__.__name__, 
            features=[f.to_dict() for f in self.features])

    def subfeature(self, other: "FeatureSpecification|list[str]", x):
        if not isinstance(other, FeatureSpecification):
            other = self.subfeature_spec(other)
        cut = self.cut_up(x)
        return other.stack(cut)

    def subfeature_spec(self, feature_list: list[str]) -> Self:
        feat_dict = {feature.name: feature for feature in self.features}
        other = self.__class__([feat_dict[name] for name in feature_list])
        return other

    @classmethod
    def from_config(cls, config:dict[str, Any]):
        assert config["type"] == cls.__name__
        loader = Loader(cls.feature_type_register)
        features = [loader(f) for f in config["features"]]
        return cls(features)

    def __int__(self):
        # auto conversion to int. can be useful
        return self.dim

