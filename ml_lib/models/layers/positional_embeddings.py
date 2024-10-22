import warnings

import math
import torch
from torch import is_floating_point, nn, Tensor

from ml_lib.misc.torch_functions import ShouldBe as should_be

class PositionalEmbeddings(nn.Module):
    half_dim: int
    max_period: int
    integer_len: int|None

    cache: Tensor|None
    angular_frequencies: Tensor
    pi_over_two: Tensor

    def __init__(self, dim, max_period=10000, *,  integer_len=None):
        assert dim % 2 == 0, 'dim must be even'
        super().__init__()
        self.half_dim = dim // 2
        self.integer_len = integer_len
        self.max_period = max_period
        self.cache = None
        self._init_frequencies()
        if integer_len is not None:
            self.cache = self._compute_cache(integer_len)
            self.register_buffer('cache', self.cache)


    @torch.no_grad
    def _init_frequencies(self):
        half_dim = self.half_dim
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, 
        )
        exponent = exponent / (half_dim)
        self.angular_frequencies= torch.exp(exponent)
        self.register_buffer('angular_frequencies', self.angular_frequencies)
        self.pi_over_two = torch.tensor(math.pi / 2)
        self.register_buffer('pi_over_two', self.pi_over_two)

    @torch.no_grad
    def _compute_cache(self, integer_len):
        return self.compute_embeddings(torch.linspace(0, 1, integer_len))

    def compute_embeddings(self, fractional_timesteps):
        """fractional timesteps are between 0 and 1"""
        *batch, = fractional_timesteps.shape

        phases = self.angular_frequencies * fractional_timesteps[..., None]   <= should_be(*batch, self.half_dim)
        phases = phases[..., None].broadcast_to(*batch, self.half_dim, 2)
        phases[..., 1] += self.pi_over_two

        phases = phases.reshape(*batch, self.half_dim * 2).contiguous()
        phases = torch.sin(phases)
        return phases


    def forward(self, timesteps):
        if torch.is_grad_enabled():
            warnings.warn("forgot to disable grad when computing embeddings")

        if torch.is_floating_point(timesteps):
            return self.compute_embeddings(timesteps)
        else:
            assert self.cache is not None
            return self.cache[timesteps]

