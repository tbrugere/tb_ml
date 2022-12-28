"""Some simple synthetic datasets of geometric shapes
"""
from math import pi
import torch
from torch.utils.data import IterableDataset
from torch import cos, sin

from .registration import register

def _sample_4Dtorus(n=1, noise=.01):
    theta = torch.rand(n) * 2 * pi
    phi = torch.rand(n) * 2 * pi
    noise = torch.normal(torch.zeros((n, 4)), noise)
    points = torch.stack([cos(theta), sin(theta), cos(phi), sin(phi)], dim=-1)
    return points + noise

def _sample_2Dcircle(n=1, noise=.01):
    theta = torch.rand(n) * 2 * pi
    noise = torch.normal(torch.zeros((n, 2)), noise)
    points = torch.stack([cos(theta), sin(theta)], dim=-1)
    return points + noise

def embed_torus_3D(t, r1 = 1, r2 = .2):
    zp1 = t[...,2]*r2 + r1
    return torch.stack([
        t[...,0]* zp1,
        t[...,1]* zp1,
        t[...,3]* r2
    ], dim=-1)

@register
class Torus4D(IterableDataset):

    size: int
    batch_size: int
    noise: float

    def __init__(self, size=10000, batch_size = 1, noise = .01):
        self.size = size
        self.batch_size  = batch_size
        self.noise = noise

    def __iter__(self):
        for _ in range(self.size // self.batch_size):
            yield _sample_4Dtorus(self.batch_size, self.noise)
            
    def __len__(self):
        return self.size // self.batch_size

@register
class Torus3D(IterableDataset):

    size: int
    batch_size: int
    noise: float

    def __init__(self, size=10000, batch_size = 1, noise = .01):
        self.size = size
        self.batch_size  = batch_size
        self.noise = noise

    def __iter__(self):
        for _ in range(self.size // self.batch_size):
            yield embed_torus_3D(_sample_4Dtorus(self.batch_size, self.noise))
            
    def __len__(self):
        return self.size // self.batch_size

@register
class Circle2D(IterableDataset):

    size: int
    batch_size: int
    noise: float

    def __init__(self, size=10000, batch_size = 1, noise = .01):
        self.size = size
        self.batch_size  = batch_size
        self.noise = noise

    def __iter__(self):
        for _ in range(self.size // self.batch_size):
            yield _sample_2Dcircle(self.batch_size, self.noise)

    def __len__(self):
        return self.size // self.batch_size
