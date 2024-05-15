from ..register import Register
from ..models import register as model_register
from ..datasets import register as dataset_register, transform_register
from .training_hooks import register as training_hook_register

loss_register = Register(object)

__all__ = ["model_register", 
           "dataset_register", 
           "transform_register", 
           "training_hook_register", 
           "loss_register"]
