"""
Transformation of datasets.

Contrarily to the ones seen in pytorch 

Basically they are actually datasets, which take other datasets as parameter.

"""
from typing import Optional, Iterator, Callable, Generic, TypeVar, NamedTuple, Union

from collections import namedtuple
from collections.abc import Sequence, Mapping
from dataclasses import dataclass, field
import itertools as it

import torch
from torch import Tensor

from ml_lib.misc.basic import deprecated
from .base_classes import Transform, Dataset, Element, Element2
from .registration import transform_register
from .datapoint import Datapoint, DictDatapoint

NamedTupleElement = TypeVar("NamedTupleElement", bound=NamedTuple)
TensorOrNamedTupleElement = TypeVar("TensorOrNamedTupleElement", bound=Union[Tensor, NamedTuple])

class SameTypeTransform(Transform[Element, Element]):
    @property
    def datatype(self) -> type[Element]:
        return self.inner.datatype

@transform_register
class CacheTransform(SameTypeTransform[Element]):
    """
    Transforms an iterable dataset, 
    into an indexable (subscriptable) dataset
    by caching n values
    """

    n: Optional[int] = None
    cached_data: list[Element]|None = None

    def __init__(self, n: Optional[int]=None):
        assert isinstance(n, int|None)
        self.n = n

    def _initialize(self):
        assert self.inner is not None
        inner_iter: Iterator[Element] = iter(self.inner)
        if self.n is not None:
            inner_iter = it.islice(inner_iter, self.n)
        
        self.cached_data = list(inner_iter)

    def __getitem__(self, i):
        assert self.cached_data is not None
        return self.cached_data[i]

    def __len__(self):
        assert self.cached_data is not None
        return len(self.cached_data)




@transform_register
class FilterTransform(SameTypeTransform[Element]):
    """Transforms a dataset by filtering out some elements

    """
    f: Callable[[Element,], bool]

    def __init__(self, f: Callable[[Element,], bool]):
        self.f = f

    def __iter__(self):
        for elem in self.inner:
            if self.f(elem):
                yield elem


@transform_register
@dataclass
class FunctionTransform(Transform[Element, Element2]):
    """Simple transform that applies a function to every element of the dataset
    """

    f: Callable[[Element,], Element2]
    datatype: type[Element2]

    def __getitem__(self, i) -> Element2:
        return self.f(self.inner[i])
    
    def __iter__(self) -> Iterator[Element2]:
        for elem in self.inner:
            yield self.f(elem)

#TODO type this correctly (need to create a return type 
#that inherits from datapoint )
# for example a "DictDatapoint" that is just a wrapper around a dict
@transform_register
class MultipleFunctionTransform(FunctionTransform[Element, DictDatapoint]):

    functions: dict[str, tuple[Callable, str|int]]
    datatype= DictDatapoint

    def __init__(self, functions):
        self.functions = functions
        self.datatype = DictDatapoint

    def f(self, inner_value: NamedTupleElement) -> NamedTupleElement:
        result = {}
        for elem_name, (func, arg) in self.functions.items():
            match arg:
                case "_":
                    mapped_elem = func(inner_value)
                case _ if isinstance(inner_value, Datapoint):
                    mapped_elem = func(inner_value.get_feature(arg))
                case int() if isinstance(inner_value, Sequence):
                    mapped_elem = func(inner_value[arg])
                case str():
                    mapped_elem = func(getattr(inner_value, arg))
                case _:
                    assert False, f"incompatible {arg=} and {inner_value=}"
            result[elem_name] = mapped_elem
        return self.datatype(result)


# Same.
@transform_register
class RenameTransform(FunctionTransform[Element, DictDatapoint]):
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
    name_map: dict[int | str, str]
    datatype: type[DictDatapoint] = DictDatapoint

    def __init__(self,  name_map):
        self.name_map = name_map
        self.datapoint_type= DictDatapoint

    def f(self, inner_value: Element) -> DictDatapoint:
        d = {}
        for field_name, key in self.name_map.items():
            match key:
                case "_":
                    d[field_name] = inner_value
                case _ if isinstance(inner_value, Datapoint):
                    d[field_name] = inner_value.get_feature(key)
                case int() if isinstance(inner_value, Sequence):
                    d[field_name] = inner_value[key]
                case str() if isinstance(inner_value, Mapping):
                    d[field_name] = inner_value[key]
                case str():
                    d[field_name] = getattr(inner_value, key)
                case _:
                    assert False, f"incompatible {key=} and {inner_value=}"
        return self.datatype(d)

# DONT USE
@deprecated
# @transform_register
class ToGPUTransform(FunctionTransform[TensorOrNamedTupleElement, TensorOrNamedTupleElement]):
    """Transforms a dataset to a dataset that sends the elements to the GPU

    """
    inner: Dataset[TensorOrNamedTupleElement]
    device: torch.device

    def __init__(self, inner: Dataset[TensorOrNamedTupleElement], device: torch.device):
        self.inner = inner
        self.device = device

    def f(self, elem):
        match elem:
            case Tensor():
                return elem.to(self.device)
            case _:
                return elem._replace(**{k: v.to(self.device) 
                                        for k, v in elem._asdict().items() if isinstance(v, Tensor)})


############################################################################
# Normalization
############################################################################


@transform_register
class TensorUniqueTransform(Transform[torch.Tensor, torch.Tensor]):
    """Deletes unique elements in a dataset
    assumes all elements in the dataset are torch tensors
    and that the dataset is sliceable (ie you can get the full tensor by using Dataset[:])
    """
    data: torch.Tensor = None

    def _initialize(self):
        self.data = torch.unique(self.inner[:], dim=0)

    def __getitem__(self, *args):
        return self.data.__getitem__(*args)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
