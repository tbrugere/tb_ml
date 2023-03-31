from typing import Callable, Optional

from torch import Tensor
from torch import nn

from ..base_classes import AutoEncoder
from ..layers.VAE import Reparameterize


class VAE(AutoEncoder):
    """Variational Autoencoder"""

    reparameterize: Reparameterize

    def __init__(self):
        super().__init__()
        self.reparameterize = Reparameterize()
        self.loss_fun = nn.MSELoss()

    def predict(self, x, return_kl=False) -> Tensor:
        z = self.encode(x)
        if return_kl:
            z, kl = self.reparameterize(z, returnkl=True)
        else:
            z = self.reparameterize(z)
        y = self.decode(z)
        if return_kl:
            return y, kl#type: ignore
        return y

    def compute_loss(self, x, loss_fun: Optional[Callable]=None):
        y, kl = self.predict(x, return_kl=True)
        if loss_fun is None:
            loss_fun = self.loss_fun
        assert loss_fun is not None, "no loss function "
        mse = loss_fun(y, x) 
        loss = mse + kl
        return loss
