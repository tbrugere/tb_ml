from typing import Sequence

import itertools as it

from torch import nn
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
