from typing import Callable, Optional

import functools as ft
import itertools as it

import torch
import torch.nn as nn

from ..environment import HasEnvironmentMixin

class Model(nn.Module, HasEnvironmentMixin):
    """
    Base class for this library’s models.
    """

    def __init__(self):
        super(nn.Module).__init__()
        super(HasEnvironmentMixin).__init__()

    def predict(self, x) -> torch.Tensor: 
        x;
        raise NotImplementedError

    def compute_loss(self, x) -> torch.Tensor:
        x;
        raise NotImplementedError("This model doesn’t have a loss function. "
                             "You should pass one to the trainer")

class Supervised(Model):

    loss_fun: Optional[Callable]
    """
    Default loss function.
    It should take two arguments, `input` and `target`
    """

    def predict(self, x) -> torch.Tensor: 
        """
        This is the function you should edit
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
        loss = loss_fun(x, gt)
        return loss


class Unsupervised(Model):
    def forward(self, x, loss=False):
        raise NotImplementedError




# class Autoencoder(Unsupervised):
#     """Base class for autoencoders"""
#     def encode(self, x) -> torch.Tensor:
#         x;
#         raise NotImplementedError
#
#     def decode(self, z) -> torch.Tensor:
#         z;
#         raise NotImplementedError
#
#     def predict(self, x):
#         z = self.encode(x)
#         y = self.decode(z)
#         return y
