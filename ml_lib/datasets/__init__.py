"""
This module contains datasets for use with geometrical models

Datasets
--------

.. autosummary::

"""

from . import transforms
from . import simple_shapes
from ..register import Loader
from .registration import register, transform_register
from .splitting import split_arrays, split_indices
from .base_classes import Dataset

load_dataset = Loader(register)
"""Dataset loader (used by the automated pipeline)"""

__all__ = ['load_dataset', 
           'register', 
           'transform_register', 
           'split_arrays', 
           'split_indices', 
           'Dataset']
