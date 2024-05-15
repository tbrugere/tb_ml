"""Some simple synthetic datasets of geometric shapes
These datasets are legacy, and I haven't updated them to use "Datapoint" classes
so try not to use too much, might break in unexpected ways
"""
from math import pi
import torch
from torch import cos, sin

from ml_lib.datasets.base_classes import Dataset
from ml_lib.datasets.registration import register
from ml_lib.datasets.feature_specification import FeatureSpecification, MSEFeature

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
class Torus4D(Dataset[torch.Tensor]):

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

    def dataset_parameters(self):
        return dict(
            feature_specification= FeatureSpecification([MSEFeature("location", 4)])
                )

@register
class Torus3D(Dataset[torch.Tensor]):

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
class Circle2D(Dataset[torch.Tensor]):

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
