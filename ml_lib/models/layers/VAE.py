"""Layers for VAE"""

import torch
from torch import Tensor
import torch.nn as nn

class Reparameterize(nn.Module):
    """Reparameterization layer

    Implements the reparameterization trick for VAEs.

    Args:
        x (torch.Tensor): (*batch, latent_dim * 2) output of ecoder layer, 
            the last dimension has the mean and logvar concatenated.
        returnkl (bool): if True, returns the KL divergence as well. Default: False
    Returns:
        Tensor: (*batch, latent_dim) reparameterized latent vector
        Tensor: (*batch,) KL divergence if returnkl is True
    """
    def forward(self, x, returnkl = False) -> Tensor | tuple[Tensor, Tensor]:
        mu, logvar = torch.tensor_split(x, 2, dim=-1)
        # both (*batch, latent_dim)
        noise = torch.randn_like(mu)
        retval =  mu + torch.exp(logvar * .5) * noise
        if not returnkl:
            return retval
        kl = - .5 * ((logvar - mu.square() - logvar.exp()).mean(dim=-1) + 1)
        return retval, kl
