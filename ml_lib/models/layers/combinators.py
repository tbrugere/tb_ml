import functools as ft
import itertools as it # pyright: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualShortcut(nn.Module):
    """Residual shortcut as used in ResNet.

    A module that adds the input to the output of another module.
    So if inner module is f, the output is x + f(x).
    
    This is useful to implement residual blocks as they were 
    originally used in resnet (and are used in a lot of modern architectures)
    """
    inner_module: nn.Module
    def __init__(self, inner_module):
        super().__init__()
        self.inner_module = inner_module
    
    def forward(self, x):
        y = self.inner_module(x)
        return x + y
    
class Repeat(nn.Sequential):
    """A module that repeats copies of a module n times.
    Takes a module fatcory as input (a function that returns a module),
    and creates n modules by calling the factory n times.
    and puts them in a sequential module.

    This is for convenience, when an architecture reads like
    " block defined above × 5"
    """
    def __init__(self, n, module_factory):
        super().__init__(*[module_factory() for _ in range(n)])


