from typing import Self
from torch.utils.data import default_collate

class Datapoint():
    """TODO"""

    def get_feature(self, name):
        return getattr(self, name)

    def asdict(self):
        raise NotImplementedError

    def to(self, device, **kwargs):
        return self.__class__(**{name: value.to(device, **kwargs) 
                                 for name, value in self.asdict().items()})

    @classmethod
    def collate(cls, datapoints: list[Self]):
        return default_collate(datapoints)



