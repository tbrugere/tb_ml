from typing import TypeVar, Generic, Iterator

from dataclasses import dataclass
# from typing import Callable, TypeAlias

import torch.utils.data as torch_data

Element = TypeVar("Element")
Element2 = TypeVar("Element2")

class Dataset(torch_data.Dataset, Generic[Element]):
    def __getitem__(self, i) -> Element:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Iterator[Element]:
        for i in range(len(self)):
            yield self[i]


@dataclass
class Transform(Dataset[Element2], Generic[Element, Element2]):
    """
    Transformation of datasets.

    Contrarily to the ones seen in pytorch 

    Basically they are actually datasets, which take other datasets as parameter.

    This base class should not be instantiated, it doesn’t do anything.

    Since this is a dataclass, the __init__() will call __post_init__(), so subclasses could
    implement that
    """
    inner: Dataset[Element]


    def __post_init__(self):
        self._initialize()

    def _initialize(self):
        pass

    def __len__(self):
        return len(self.inner)#type: ignore

# class IterableTransform(Transform, IterableDataset):
#     pass
