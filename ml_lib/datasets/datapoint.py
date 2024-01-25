from typing import Self
from torch.utils.data import default_collate

class Datapoint():
    """TODO"""

    def get_feature(self, name):
        raise NotImplementedError

    def asdict(self):
        raise NotImplementedError

    @classmethod
    def collate(cls, datapoints: list[Self]):
        return default_collate(datapoints)



