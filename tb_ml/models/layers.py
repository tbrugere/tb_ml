import functools as ft
import itertools as it # pyright: ignore
from inspect import signature

import torch
import torch.nn as nn



def MLP_rectangular(n_layers, dim_in, dim_hidden, dim_out, activation=nn.ELU, activation_out=None):
    if activation_out is None:
        activation_out = activation
    if "dim" in signature(activation).parameters:
         #trick to pass dim if needed
        activation = ft.partial(activation, dim=-1)
    if "dim" in signature(activation_out).parameters:
         #trick to pass dim if needed
        activation_out = ft.partial(activation_out, dim=-1)

    layers = []
    n_in = dim_in
    n_out = dim_hidden
    for i in range(n_layers - 1):
        layers.append(nn.Linear(n_in, n_out))
        layers.append(activation())
        n_in = n_out
    n_out = dim_out
    layers.append(nn.Linear(n_in, n_out))
    layers.append(activation_out())
    return nn.Sequential(*layers)

class MLP(nn.Sequential):
    def __init__(self, *dims: int, batchnorm=True, end_activation=False, activation=nn.ReLU):
        super().__init__()
        for i, input_dim, output_dim in zip(it.count(), dims, dims[1:]):
            self.add_module(f"linear_{i}", nn.Linear(input_dim, output_dim))
            if i == len(dims)-2 and not end_activation:
                continue #skip batchnorm and activation at last layer
            if batchnorm:
                self.add_module(f"norm_{i}", nn.BatchNorm1d(output_dim))
            self.add_module(f"activation_{i}", activation())

class DoubleSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(DoubleSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.log_softmax = nn.LogSoftmax(dim=dim)


    def forward(self, x):
        return torch.stack((self.softmax(x), self.log_softmax(x)))


