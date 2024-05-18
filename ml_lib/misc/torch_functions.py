from typing import Sequence

import itertools as it

import torch
from torch import Tensor, nn

from .basic import eventually_tuple
#-----------------------------------------------------------
# Pytorch stuff
#-----------------------------------------------------------

def freeze_model(m: nn.Module):
    for param in m.parameters():
        param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_gradient(model):
    return sum(p.grad.numel() for p in model.parameters() if p.requires_grad)

def broadcastable(*shapes: Sequence[int]):
    """
    Returns whether the given shapes are broadcastable.
    """
    from . import all_equal
    iterator = it.zip_longest(*map(reversed, shapes), fillvalue=1)
    for sizes in iterator:
        not1 = [s for s in sizes if s != 1]
        if not all_equal(*not1):
            return False
    return True

def get_default_device():
    return torch.Tensor().device

def num_parameters_tree(model: torch.nn.Module, depth=3, 
                        human_readable=True, return_total = False, 
                        indent_by=4, model_name= None, 
                        skip_empty=False, 
                        gradient: bool=False):
    """Warning: This function does not handle the same parameters being used by different submodules (they WILL be counted twice.) 
    Use count_parameters directly to check

    Prints submodules as 



    Args:
        model: The model 
        depth: The max depth to recurse into
        gradient: if true, then count the size of the gradients instead of the sizes of the parameters. useful for debugging. 
                 Only defined if the gradients were computed!

    """
    from .basic import human_readable as hr
    from textwrap import indent
    from io import StringIO
    if not human_readable: hr = lambda x: x
    model_name = model_name or model.__class__.__name__
    total = 0
    hr_string = ""
    if depth == 0:
        n_parameters = count_parameters(model) if not gradient else count_parameters_gradient(model)
        hr_string = f"- ({model_name}): {hr(n_parameters)}"
        total = n_parameters
    else:
        assert depth > 0, "depth cannot be negative"
        module_strings = []
        for name, module in model.named_children():
            module_string, module_size = num_parameters_tree(
                    module, depth=depth-1, model_name=name, 
                    human_readable=human_readable, return_total=True, 
                    indent_by=indent_by, skip_empty=skip_empty)
            total += module_size
            if module_size == 0 and skip_empty:
                continue
            module_strings.append(indent(module_string, prefix=indent_by * " "))
        hr_string = "\n".join([
            f"- ({model_name}): {hr(total)}", 
            *module_strings
            ])

    eventually_return_total = [total] if return_total else []

    return eventually_tuple(hr_string, *eventually_return_total)


def move_batch_to(batch, device, non_blocking=False, ignore_failure=False):
    match batch:
        case batch if hasattr(batch, "to"):
            return batch.to(device, non_blocking=non_blocking)
        case batch if hasattr(batch, "_asdict"):
            t_ = type(batch)
            return t_(**{key: value.to(device, non_blocking=non_blocking) 
                         for key, value in batch._asdict().items()})
        case dict():
            return {key: value.to(device, non_blocking=non_blocking) 
                    for key, value in batch.items()}
        case _ if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=non_blocking)
        case (*seq,):
            return type(seq)(i.to(device, non_blocking=non_blocking) 
                             for i in seq)
        case _ if not ignore_failure:
            raise ValueError(f"Couldn't find how to move object {batch} to device {device}")

class ShouldBe():
    """
    Nice little class that allows checking shapes of torch tensors without too much verbosity / distraction.

    Basically I often end up writing code where I annotate with the shape each tensor should have

    such as this:

    .. code-block:: python

        def forward(self, x, y):
            b, n, dx = x.shape
            b_, m, dy = y.shape
            assert b == b_
            assert dx == self.dx
            assert dy == self.dy


            key  = self.mlp_x(x) # b, n, hidden
            query = self.mlp_y(y) # b, m, hidden
            logits = (key[:, :, None, :] * query[:, None, :, :]).sum(dim=-1) # b, n, m

           ...

    And I would like those annotations to be translated into asserts, but it would be very heavy


    .. code-block:: python

        def forward(self, x, y):
            b, n, dx = x.shape
            b_, m, dy = y.shape
            assert b == b_
            assert dx == self.dx
            assert dy == self.dy


            key  = self.mlp_x(x) 
            assert key.shape == b, n, hidden
            query = self.mlp_y(y) 
            assert query.shape == b, m, hidden
            logits = (key[:, :, None, :] * query[:, None, :, :]).sum(dim=-1) 
            assert logits.shape == b, n, m
           ...
            
    It really clutters the code (double the length)

    Instead now I can write

    .. code-block:: python

        def forward(self, x, y):
            from ml_lib.misc.torch_functions import ShouldBe as should_be
            b, n, dx = x.shape
            b_, m, dy = y.shape
            assert b == b_
            assert dx == self.dx
            assert dy == self.dy


            key  = self.mlp_x(x)                        <= should_be(b, n, hidden)
            query = self.mlp_y(y)                       <= should_be(b, m, hidden)
            logits = (key[:, :, None, :] * query[:, None, :, :]).sum(dim=-1) <= should_be(b, n, m)

           ...

    """
    shape: Sequence[int]
    dtype: torch.dtype|None=None
    device: torch.device|None = None

    def __init__(self, *shape, dtype=None, device=None):
        self.shape = shape
        if dtype is not None: self.dtype = dtype
        else: self.dtype=None
        if device is not None: self.device = torch.device(device)
        else: self.device =None

    def __invert__(self): 
        """ so that you can write 
            ``tensor<=~~~~~~~~~~~~~~~~~~~~~~~~Shouldbe(n, m)`` This is probably not very """
        return self


    def __ge__(self, other):
        """called as ``tensor <= ShouldBe(n, m)``"""
        if not isinstance(other, Tensor): return NotImplemented

        if self.shape != other.shape:
            raise ValueError(f"Shape check failed: expected {self.shape}, got {other.shape}")
        if self.dtype is not None and self.dtype != other.dtype:
            raise ValueError(f"Dtype check failed: expected {self.dtype}, got {other.dtype}")
        if self.device is not None and self.device != other.device:
            raise ValueError(f"Dtype check failed: expected {self.device}, got {other.device}")

        return other


    def __gt__(self, other):
        """called as ``tensor < ShouldBe(n, m)``"""
        return self.__ge__(other)
