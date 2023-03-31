from typing import Callable, Optional

import functools as ft
import itertools as it
from types import FunctionType

import torch
import torch.nn as nn

from ..environment import HasEnvironmentMixin

def _get_default_device():
    return torch.Tensor().device

class ModelMeta(type):
    """
    Metaclass for models. 
    Ensures that all the model’s function runs in the model’s context manager.
    """
    def __new__(cls, name, bases, class_dict):
        new_class_dict = {}
        for attr_name, attr in class_dict.items():
            if isinstance(attr, FunctionType):
                attr = cls.use_model_context(attr)
            new_class_dict[attr_name] = attr
        if "device" not in new_class_dict:
            new_class_dict["device"] = _get_default_device()
        return type.__new__(cls, name, bases, new_class_dict)

    @staticmethod
    def use_model_context(f):
        @ft.wraps(f)
        def wrapped(self, *args, **kwargs):
            device = self.device
            with device:
                return f(self, *args, **kwargs)
        return wrapped

class Model(nn.Module, HasEnvironmentMixin, metaclass=ModelMeta):
    """
    Base class for this library’s models.
    Got a bunch of sugar including:
        - keeping track of the device the model is on
        - ensuring that all the model’s function run on the correct device by default
        - somewhere to place loss functions
        - some useful functions
    """

    device: torch.device

    def __init__(self):
        super(nn.Module, self).__init__()
        super(HasEnvironmentMixin, self).__init__()
        self.device = torch.device("cpu")

    def predict(self, x) -> torch.Tensor: 
        x;
        raise NotImplementedError

    def compute_loss(self, x) -> torch.Tensor:
        x;
        raise NotImplementedError("This model doesn’t have a loss function. "
                             "You should pass one to the trainer")

    def _apply(self, fn):
        # keeps the device in sync with the model
        # https://stackoverflow.com/a/70593357/4948719
        super()._apply(fn)
        self.device = fn(self.device)
        return self

    def num_parameters(self, trainable_only: bool = False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

class Supervised(Model):

    loss_fun: Optional[Callable]
    """
    Default loss function.
    It should take two arguments, `input` and `target`
    """

    def predict(self, x) -> torch.Tensor: 
        """
        This is the function you should override to implement your model.
        """
        x;
        raise NotImplementedError

    def forward(self, x) -> torch.Tensor: 
        y = self.predict(x)
        return y

    def compute_loss(self, x, gt, loss_fun: Optional[Callable]=None):
        y = self(x)
        if loss_fun is None:
            loss_fun = self.loss_fun
        assert loss_fun is not None, "no loss function "
        loss = loss_fun(y, gt)
        return loss


class Unsupervised(Model):
    pass


class AutoEncoder(Unsupervised):
    """Base class for autoencoders"""

    loss_fun: Optional[Callable]
    """
    Default loss function.
    It should take two arguments, `input` and `target`
    """

    def encode(self, x) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x) -> torch.Tensor:
        return self.predict(x)

    def predict(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y

    def compute_loss(self, x, loss_fun: Optional[Callable]=None):
        y = self(x)
        if loss_fun is None:
            loss_fun = self.loss_fun
        assert loss_fun is not None, "no loss function "
        loss = loss_fun(y, x) 
        return loss
