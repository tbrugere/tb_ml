# import collections
# import functools as ft

import torch.nn as nn

from ..register import Register

register : Register[nn.Module]= Register(nn.Module, name_field="model_name")

