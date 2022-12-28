"""
Transformation of datasets.

Contrarily to the ones seen in pytorch 

Basically they are actually datasets, which take other datasets as parameter.

"""
from typing import Optional, Iterator

from collections import namedtuple
from collections.abc import Sequence, Mapping
from dataclasses import dataclass, field
import itertools as it

import torch

from .base_classes import Transform, Dataset, Element, Element2
from .registration import transform_register

@transform_register
@dataclass
class CacheTransform(Transform[Element, Element]):
    """
    Transforms an iterable dataset, 
    into an indexable (subscriptable) dataset
    by caching n values
    """

    inner: Dataset[Element]
    cached_data: list[Element]

    def __init__(self, inner: Dataset[Element], n: Optional[int]=None):
        self.inner = inner
        inner_iter: Iterator[Element] = iter(inner)
        if n is not None:
            inner_iter = it.islice(inner_iter, n)
        
        self.cached_data = list(inner_iter)

    def __getitem__(self, i):
        return self.cached_data[i]

    def __len__(self):
        return len(self.cached_data)



@transform_register
@dataclass
class RenameTransform(Transform[Element, Element2]):
    """Rename the outputs of a dataset

    Converts a dataset whose elements are tuples, dicts, or namedtuples to
    dataset whose elements are namedtuples, 
    using a name correspondance `named_map`.

    More precisely, 

    If 
    ```python
    named_map = {
        input_name1: "output_name1", 
        input_name2: "output_name2",
        input_name3: "output_name3"
    }
    ```

    and `dataset` is the input dataset, whose elements are namedtuples, 
    dicts, or tuples

    Then `t = RenameTransform(dataset, name_map)[i]` is a namedtuple, 
    with fields "output_name1", "output_name2" and "output_name3"

    whose values are resp. `dataset[i][input_name1], dataset[i][input_name2], dataset[i][input_name3]`
    (or the fields if `dataset[i]` is a `namedtuple`)

    The input names can be either 
    - The special name "_", which is mapped to the entire input `dataset[i]`
    - A string if dataset’s elements are mappings 
        (in which case __getitem__ is used)
    - An int if dataset’s elements are sequences
        (in which case __getitem__ is used)
    - A string if datasets’s elements are objects 
        (in which case __getattr__ is used)

    """
    inner: Dataset[Element]
    name_map: dict[int | str, str]
    datapoint_type: type[Element2] = field(init=False)

    def __post_init__(self):
        self.element_type= namedtuple("datapoint", self.name_map.values())

    def __getitem__(self, index) -> Element2:
        inner_value: Element = self.inner[index]
        d = {}
        for key, field_name in self.name_map.items():
            match key:
                case "_":
                    d[field_name] = inner_value
                case int() if isinstance(inner_value, Sequence):
                    d[field_name] = inner_value[key]
                case str() if isinstance(inner_value, Mapping):
                    d[field_name] = inner_value[key]
                case str():
                    d[field_name] = getattr(inner_value, key)
                case _:
                    assert False, f"incompatible {key=} and {inner_value=}"

        return self.datapoint_type(**d)


############################################################################
# Normalization
############################################################################


@transform_register
class TensorUniqueTransform(Transform[torch.Tensor, torch.Tensor]):
    """Deletes unique elements in a dataset
    assumes all elements in the dataset are torch tensors
    and that the dataset is sliceable (ie you can get the full tensor by using Dataset[:])
    """
    inner: Dataset[torch.Tensor]

    data: torch.Tensor

    def _initialize(self):
        self.data = torch.unique(self.inner[:], dim=0)

    def __getitem__(self, *args):
        return self.data.__getitem__(*args)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
