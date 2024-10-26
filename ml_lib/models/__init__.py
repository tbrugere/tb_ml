"""
This module contains the base classes for models, as well as some layer implementations for neural networks.

The most important class here the :class:`Model` class. Every model should inherit from this class. (but not layers which are classic :class:`torch.nn.Module`)

Base usage:

.. code-block:: python
   from typing import Literal
   from ml_lib.models import Model, Hyperparameter
   from ml_lib.models.layers import MLP
   class MyModel(ml_lib.models.Model):
       n_layers: Hyperparameter[int]
       input_dim: Hyperparameter[int]
       output_dim: Hyperparameter[int]
       hidden_dim: Hyperparameter[int] = 512
       loss: Hyperparameter[Literal["mse", "cross_entropy"]] = "mse

       inner: MLP
       loss_function: torch.nn.MSELoss

       def __setup__(self):
           # construct the model from hyperparameter values
           # in this function, the hyperparameter values are already populated
           self.inner = MLP(input_dim, *[hidden_dim]*n_layers - 1, output_dim)

       def compute_loss(self, x, y):
           z = self.inner(x)
           return self.loss_function(z, y)

    model = MyModel(n_layers=3, input_dim=10, output_dim=1)

Notice the absence of a constructor: one should already be provided by :class:`Model`, that inputs the hyperparameter as keyword-only arguments, calls ``nn.Module.__init__`` 
**checks types for all the hyperparameters**

Using this class will have several benefits out of the box, apart from not forgetting to call ``super().__init__``, and it working out of the box with :class:`ml_lib.pipeline.Trainer`: 

The model has a well-defined device, accessible with ``model.device`` (which changes when the model is moved to a different device.)
All methods of the module are automatically run as if they were in the context manager
.. code-block:: python
   with self.device:
       ...
which means all tensors are created on the correct device by default.
The arguments to functions will also be checked to see if they are on model device.

(all these behaviours can be deactivated)
"""

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
