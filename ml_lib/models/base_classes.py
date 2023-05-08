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
            if isinstance(attr, FunctionType) and attr_name != "__init__":
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

    _dummy_param: nn.Parameter

    def __init__(self):
        nn.Module.__init__(self)
        HasEnvironmentMixin.__init__(self)
        # self.device = _get_default_device()
        self._dummy_param = nn.Parameter()

    def predict(self, x) -> torch.Tensor: 
        """Procedure used at inference time to compute the output"""
        raise NotImplementedError

    def compute_loss(self, x) -> torch.Tensor:
        """Procedure used at training time to compute the loss"""
        raise NotImplementedError("This model doesn’t have a loss function. "
                             "You should pass one to the trainer")

    def forward(self, x) -> torch.Tensor: 
        return self.predict(x)

    @property
    def device(self):
        return self._dummy_param.device

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
    encoder: Callable
    decoder: Callable
    """
    Default loss function.
    It should take two arguments, `input` and `target`
    """

    def encode(self, x) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)

    def recognition_loss(self, x, z, loss_fun: Optional[Callable]=None):
        if loss_fun is None: loss_fun = self.loss_fun
        assert loss_fun is not None, "no loss function "
        y = self.decode(z)
        return loss_fun(y, x)

    def predict(self, x):
        return self.encode(x)

    def compute_loss(self, x, loss_fun: Optional[Callable]=None, **kwargs):
        z = self.encode(x)
        loss = self.recognition_loss(x, z, loss_fun, **kwars)
        return loss

class Head(nn.Module):
    """Head modules. These modules are used to change output types"""

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self, x, gt):
        raise NotImplementedError

class BodyHeadSupervised(Supervised):
    """Base class for models with a body and a head.
    This is a pattern I use often which is why I’m defining it here

    """

    body: nn.Module
    head: Head

    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head

    def predict(self, x):
        return self.head(self.body(x))

    def compute_loss(self, x, gt, **kwargs):
        y = self.body(x)
        loss = self.head.compute_loss(y, gt, **kwargs)
        return loss
