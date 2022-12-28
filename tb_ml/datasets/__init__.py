"""
This module contains datasets for use with geometrical models

Datasets
--------

.. autosummary::

"""

from . import transforms
from ..register import Loader
from .registration import register, transform_register

load_dataset = Loader(register)
"""Dataset loader (used by the automated pipeline)"""
