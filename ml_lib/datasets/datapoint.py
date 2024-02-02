from typing import Self
from torch.utils.data import default_collate

class Datapoint():

    def get_feature(self, name):
        return getattr(self, name)

    def asdict(self):                                                              
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def to(self, device, **kwargs):
        return self.__class__(**{name: value.to(device, **kwargs) 
                                 for name, value in self.asdict().items()})

    @classmethod
    def collate(cls, datapoints: list[Self]):
        return default_collate(datapoints)



class DictDatapoint():

    data: dict

    def __init__(self, data):
        self.data = data

    def get_feature(self, name):
        return self.data[name]

    def asdict(self):                                                              
        return return self.data

    def to(self, device, **kwargs):
        return self.__class__(
                {name: value.to(device, **kwargs) 
                for name, value in self.asdict().items()})

    @classmethod
    def collate(cls, datapoints: list[Self]):
        return cls(default_collate(d.data for d in datapoints)) # I think default_collate accepts dicts
