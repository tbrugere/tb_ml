from typing import Sequence

import itertools as it

import torch
from torch import nn

from .basic import eventually_tuple
#-----------------------------------------------------------
# Pytorch stuff
#-----------------------------------------------------------

def freeze_model(m: nn.Module):
    for param in m.parameters():
        param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
                        skip_empty=False):
    """Warning: This function does not handle the same parameters being used by different submodules (they WILL be counted twice.) 
    Use count_parameters directly to check

    Prints submodules as 



    Args:
        model: The model 
        depth: The max depth to recurse into

    """
    from .basic import human_readable as hr
    from textwrap import indent
    from io import StringIO
    if not human_readable: hr = lambda x: x
    model_name = model_name or model.__class__.__name__
    total = 0
    hr_string = ""
    if depth == 0:
        n_parameters = count_parameters(model)
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

