from typing import Callable, Optional

from torch import Tensor
from torch import nn

from ..base_classes import AutoEncoder
from ..layers.VAE import Reparameterize


class VAE(AutoEncoder):
    """Variational Autoencoder"""

    reparameterize: Reparameterize
    encoder: Callable
    decoder: Callable

    def __init__(self):
        super().__init__()
        self.reparameterize = Reparameterize()
        # self.loss_fun = nn.MSELoss()

    def predict(self, x, return_kl=False) -> Tensor:
        z = self.encode(x)
        if return_kl:
            z, kl = self.reparameterize(z, returnkl=True)
        else:
            z = self.reparameterize(z, returnkl = False)
        if return_kl:
            return z, kl#type: ignore
        return z

    def compute_loss(self, x, loss_fun: Optional[Callable]=None, kl_coef=1.):
        z, kl = self.predict(x, return_kl=True)
        mse = self.recognition_loss(x, z, loss_fun=loss_fun)
        loss = mse + kl_coef * kl
        return loss
