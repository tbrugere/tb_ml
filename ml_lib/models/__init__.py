from .base_classes import Model, Supervised, Unsupervised, Hyperparameter

from ..register import Loader
from .registration import register

load_model = Loader(register)

__all__ = [
    "Model", 
    "Supervised", 
    "Unsupervised",
    "load_model",
    "register", 
    "Hyperparameter"
]
