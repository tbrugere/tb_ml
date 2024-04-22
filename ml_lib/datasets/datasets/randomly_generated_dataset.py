from typing import TypeVar

from numpy.random import Generator, default_rng

from ..base_classes import Dataset
from ..datapoint import Datapoint



PointType = TypeVar("PointType", bound=Datapoint)


class GeneratedDataset(Dataset[PointType]):
    """Simple base class to create a dataset of randomly generated points, 
    that is 
    - reproducible (same dataset given same parameter)
    - indexable (*not* an iterabledataset)
    - safe (i believe)
    
    Args:
        length: the length of the dataset.
        seed: the seed (defaults to ``42`` because the question was "which seed should I choose" all along)
        which: optional argument used to differentiate between train / eval / test dataset. Can be anything. If you enter different values in "which" you should end up with different datasets (even with the same seed). You could also just use a different seed though, this is totally optional, just for convenience.
    """

    seed: int = 42
    length: int

    sub_dataset: int

    def __init__(self, length: int, seed: int = 42, which: int|str = "train"):
        self.seed = seed
        self.length = length
        self.sub_dataset_seed = hash(which)

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> PointType:
        rng = default_rng((self.seed, self.sub_dataset, i))
        return self.generate_item(rng)

    def generate_item(self, rng: Generator) -> PointType:
        """Should use the random generator ``rng`` to generate a datapoint"""
        del rng
        raise NotImplementedError("This is a base class")
