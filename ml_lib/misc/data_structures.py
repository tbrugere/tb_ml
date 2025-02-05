from typing import Generic, TypeVar, Self, Callable, Iterator, overload, Literal, Any, TypeAlias

T = TypeVar("T")

class SingletonMeta(type):
    """Metaclass for singletons
    stolen from https://stackoverflow.com/a/6798042/4948719 
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class NotSpecified(metaclass=SingletonMeta):
    """Used to specify that something is not specified"""

class Maybe(Generic[T]):
    """Proper optional type

    *m will eventually be the contained value

    Note that nicely, this can contain ``None``
    """

    is_empty: bool
    _value: T|None

    def __init__(self, value: T|NotSpecified=NotSpecified()):
        if value is NotSpecified():
            self.is_empty=True
            self._value = None
            return
        else:
            self.is_empty = False
            self._value = value #type:ignore


    @property
    def value(self) -> T:
        return self.get()

    @value.setter
    def value(self, value: T):
        self.is_empty = False
        self._value = value


    def get(self, default=NotSpecified()) -> T:
        if self.is_empty == False:
            assert self._value is not None
            return self._value
        if isinstance(default, NotSpecified):
            raise ValueError("used get with no default on empty Maybe object")
        return default

    def map(self, f: Callable[[T], T]) -> Self:
        if self.is_empty:
            return self
        return self.__class__(f(self.value))

    def bind(self,f: Callable[[T], Self] ) -> Self:
        if self.is_empty:
            return self
        return f(self.value)

    def __iter__(self) -> Iterator[T]:
        if self.is_empty: return
        yield self.value

    def __repr__(self):
        if self.is_empty:
            return "Maybe()"
        return f"Maybe({self.value})"



############### dict functions

def merge_dicts(*dicts: dict) -> dict:
    """merge dictionaries

    merges the dicts, giving precedence to the last one, RECURSIVELY

    I use this to merge configurations 
    (so say you’ve loaded a default config with all fields, 
     and a user config with some modified fields, you can merge them)

    Args:
        dicts: dictionaries

    Returns:
        dict: a dictionary containing all the keys in the dictionaries, with the value of the last argument containing that key
    """
    merged = {}
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict) and key in merged:
                value = merge_dicts(merged[key], value)
            merged[key] = value

    return merged


def unwrap_dict(d: dict, *keys):
    """[TODO:summary]

    asserts that the keys in d are the ones in the keys list, 
    and returns the elements in that order

    Args:
        d: [TODO:description]
    """
    assert set(d.keys()) == set(keys), f"wrong set of keys, got {d.keys()}, expected {keys}"
    return [d[key] for key in keys]


    

############### Union-Find

class UnionFind(list):
    """Union-find data structure with path compression"""
    def __init__(self, n):
        super().__init__(range(n))

    def _get_parent(self, k):
        return super().__getitem__(k)

    def __getitem__(self, k):
        pk = self._get_parent(k)
        if pk == k:
            return pk
        cl = self[pk]
        super().__setitem__(k, cl)
        return cl

    def __setitem__(self, k, l):
        super().__setitem__(k, self[l])

    def union(self, k, l):
        """merges the class of k into that of l"""
        self[self[k]] = self[l]

class UnionFindNoCompression(list):
    """Union-find data structure without path compression"""
    def __init__(self, n):
        super().__init__(range(n))

    def _get_parent(self, k):
        return super().__getitem__(k)

    def __getitem__(self, k):
        pk = self._get_parent(k)
        if pk == k:
            return pk
        cl = self[pk]
        #super().__setitem__(k, cl)#path compression
        return cl

    def __setitem__(self, k, l):
        super().__setitem__(k, l)

    def union(self, k, l):
        """merges the class of k into that of l"""
        self[self[k]] = l

    def reroot(self, k, _previous=None):
        """changes the representant of k’s class to k"""
        if _previous is None:
            _previous = k
        p = self._get_parent(k)
        super().__setitem__(k, _previous)
        if p == k:
            return
        self.reroot(p, _previous=k)

def check_parameters(required, provided, 
                     missing_message="missing parameters",
                     extra_message="extra parameters", 
                     wrong_message="wrong parameters"):
    from ml_lib.misc.matchers import EmptySet
    required = set(required)
    provided = set(provided)

    match (required-provided, provided-required):
        case EmptySet(), EmptySet():
            pass
        case _, EmptySet() if allow_missing:
            pass
        case missing, EmptySet():
            raise ValueError(f"{missing_message} : {missing}")
        case EmptySet(), unknown:
            raise ValueError(f"{extra_message}: {unknown}")
        case _, _:
            raise ValueError(f"{wrong_message}: \n"
                             f"expected: {required}\n"
                             f"provided (including defaults and inferred: {provided})"                                 )

