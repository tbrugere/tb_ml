from typing import TypeVar, Generic, Iterator, ClassVar, overload, NoReturn

from dataclasses import dataclass
# from typing import Callable, TypeAlias

import torch.utils.data as torch_data
from .datapoint import Datapoint

Element = TypeVar("Element", bound=Datapoint)
Element2 = TypeVar("Element2", bound=Datapoint)

class Dataset(torch_data.Dataset, Generic[Element]):
    datatype: type[Element]

    def __getitem__(self, i) -> Element:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Iterator[Element]:
        for i in range(len(self)):
            yield self[i]

    def collate(self, data: list[Element]):
        return self.datatype.collate(data)

    def dataset_parameters(self):
        """Parameters that can be inferred from the dataset. 
        This is given to models. 
        It can be eg. feature_size or feature_specs"""
        return dict()

class Transform(Dataset[Element2], Generic[Element, Element2]):
    """
    Transformation of datasets.

    Contrarily to the ones seen in pytorch 

    Basically they are actually datasets, which take other datasets as parameter.

    This base class should not be instantiated, it doesn’t do anything.

    Since this is a dataclass, the __init__() will call __post_init__(), so subclasses could
    implement that
    """
    _inner: Dataset[Element]|None = None

    def __init__(self):
        self._inner = None

    def _initialize(self):
        pass

    def __len__(self):
        return len(self._inner)#type: ignore

    def __call__(self, dataset):
        if self._inner is not None:
            #TODO copy
            raise ValueError("Already applied transform cannot be reapplied for now")
        self._inner = dataset
        self._initialize()
        return self

    @property
    def inner(self) -> Dataset[Element]:
        if self._inner is None:
            raise ValueError(f"Transform {self} needs to be applied to a dataset before using it")
        return self._inner

    def dataset_parameters(self):
        """The default is simply to pass through"""
        return self.inner.dataset_parameters()

# class IterableTransform(Transform, IterableDataset):
#     pass
