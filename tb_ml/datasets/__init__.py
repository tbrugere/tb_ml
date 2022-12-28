"""
This module contains datasets for use with geometrical models

Datasets
--------

.. autosummary::
    Circle2D
    Torus3D
    Torus4D
    AIMShapeDataset

"""

from .simple_shapes import Torus4D, Torus3D, Circle2D
from .aim_shape_loader import AIMShapeDataset

from ..register import Loader
from .registration import register

load_dataset = Loader(register)
"""Dataset loader (used by the automated pipeline)"""
