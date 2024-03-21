"""
This module contains datasets for use with geometrical models

Datasets
--------

.. autosummary::

"""

from . import transforms
from . import datasets
from ..register import Loader
from .registration import register, transform_register
from .splitting import split_arrays, split_indices
from .base_classes import Dataset, Transform

load_dataset = Loader(register)
load_transform = Loader(transform_register)
"""Dataset loader (used by the automated pipeline)"""

__all__ = ['load_dataset', 
           'load_transform', 
           'register', 
           'transform_register', 
           'split_arrays', 
           'split_indices', 
           'Dataset']
