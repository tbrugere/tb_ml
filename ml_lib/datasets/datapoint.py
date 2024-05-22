from typing import Self
from torch.utils.data import default_collate
import dataclasses

from ml_lib.misc.torch_functions import move_batch_to

class Datapoint():

    def get_feature(self, name):
        return getattr(self, name)

    def asdict(self):                                                              
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def to(self, device, **kwargs):
        return self.__class__(**{name: value.to(device, **kwargs) 
                                 for name, value in self.asdict().items()})

    def __repr__(self):
        args = ", ".join(f"{name}={item}" for name, item in self.asdict().items())
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def collate(cls, datapoints: list[Self]):
        return default_collate(datapoints)

    def __getstate__(self):
        return self.asdict()

    def __setstate__(self, d):
        for name, value in d.items():
            setattr(self, name, value)
        


class DictDatapoint(Datapoint):

    data: dict

    def __init__(self, data):
        self.data = data

    def get_feature(self, name):
        return self.data[name]

    def asdict(self):                                                              
        return self.data

    def to(self, device, **kwargs):
        return self.__class__(
                {name: move_batch_to(value, device, ignore_failure=True,  **kwargs) 
                for name, value in self.asdict().items()})

    def __getattr__(self, attr):
        assert attr != "data", "data name cannot be accessed through getattr (because it is reserved) for the underlying dict. This should not happen"
        try:
            return self.get_feature(attr)
        except KeyError as e:
            raise AttributeError from e

    @classmethod
    def collate(cls, datapoints: list[Self]):
        return cls(default_collate([d.data for d in datapoints])) # I think default_collate accepts dicts

    def __setstate__(self, d):
        self.data = d
