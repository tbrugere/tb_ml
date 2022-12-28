"""
This module contains datasets for use with geometrical models

Datasets
--------

.. autosummary::

"""

from ..register import Loader
from .registration import register

load_dataset = Loader(register)
"""Dataset loader (used by the automated pipeline)"""
