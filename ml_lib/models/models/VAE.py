from typing import Callable, Optional

import torch
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
        if loss_fun is None: loss_fun = self.loss_fun
        z, kl = self.predict(x, return_kl=True)
        mse = self.recognition_loss(x, z, loss_fun=loss_fun)
        loss = mse + kl_coef * kl.mean()
        return loss

    def sample(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim)
        return self.decode(z)

    _latent_dim: Optional[int] = None

    @property
    def latent_dim(self):
        if self._latent_dim is not None:
            return self._latent_dim
        return self.encoder.out_features

    @latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value

