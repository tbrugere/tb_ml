"""random stuff that should be in the standard lib
"""

from typing import Sequence, Final, Iterable, TypeVar, overload, TypeVarTuple, Unpack, Callable, TypeAlias

from contextlib import contextmanager
import functools as ft
import os
from math import log
import warnings
from pathlib import Path
from .data_structures import NotSpecified

T = TypeVar("T")
U = TypeVar("U")
Tuple_T = TypeVarTuple("Tuple_T")

Factory: TypeAlias = Callable[[], T]

@contextmanager
def cwd(path):
    """Context manager to change the working directory, 
    gotten from https://stackoverflow.com/a/37996581/4948719"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def find_file(possible_paths: Iterable[Path], raise_error: str | None=None):
    """returns the first path that exists, or None if none exist"""
    for path in possible_paths:
        if path.exists():
            return path
    if raise_error is not None:
        raise FileNotFoundError(raise_error)
    return None

    
def deprecated(func):
    """
    shamelessly copy-pasted from https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically

    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @ft.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

def auto_repr(*fields: str):
    """
    decorator.

    Creates an automatic __repr__ function for a class. useful for debugging since printing
    Args:
        fields: the fields to display in that repr
    """

    def repr(self, fields: Sequence[str]):
        return f"""{self.__class__.__name__}({
            ', '.join(f"{field}= {getattr(self, field)}" for field in fields)
            })"""

    def decorator(cls):
        cls.__repr__ = ft.partialmethod(repr, fields=fields)
        return cls

    return decorator

def all_equal(*args):
    """
    tests whether all the passed arguments are equal. 
    useful for checking dimensions for a lot of vectors for ex
    """
    match args:
        case ():
            return True
        case (x0, *rest):
            return all(i == x0 for i in rest)

def human_readable(num: int|float, suffix="", precision=2):
    """
    return a human readable representation of a number
    Args:
        num: the number to represent
        suffix: a suffix to add to the number (for ex "B" for bytes)
    """
    if num == 0: return f"0{suffix}"
    isneg = num < 0; num = abs(num)
    suffixes:Final[list[str]] = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    suff_n = int(log(num, 1000))
    if suff_n >= len(suffixes): suff_n = len(suffixes) - 1
    suff = suffixes[suff_n]
    num /= 1000**suff_n

    return f"{'-' if isneg else ''}{num:.{precision}f}{suff}{suffix}"

def caller_name(x):
    if hasattr(x, "get_model_type"):
        return x.get_model_type()
    if hasattr(x, "__name__"):
        return x.__name__
    if hasattr(x, "__class__"):
        return x.__class__.__name__
    return str(x)

# @overload
# def eventually_tuple(arg: T, /) -> T:
#     ...
#
# @overload
# def eventually_tuple(arg: T, arg2: U, /,  *args: Unpack[Tuple_T]) -> tuple[T, U, Unpack[Tuple_T]]:
#     ...

def eventually_tuple(*arguments):
    """If given one argument, returns that argument
    Otherwise returns that
    """
    if len(arguments) == 1:
        arg, = arguments
        return arg
    return tuple(arguments)

@overload
def fill_default(value: T|NotSpecified, *, 
                 env_variable: str|NotSpecified=NotSpecified(), 
                 default_value: T|NotSpecified=NotSpecified(),
                 default_factory: Factory[T]|NotSpecified=NotSpecified(),
                 name: str|None = None, 
                 type_convert: Callable[[str], T] 
                 ) -> T:
    ...

@overload 
def fill_default(value: T|NotSpecified, *, 
                 env_variable: str|NotSpecified=NotSpecified(), 
                 default_value: T|NotSpecified=NotSpecified(),
                 default_factory: Factory[T]|NotSpecified=NotSpecified(),
                 name: str|None = None, 
                 type_convert: NotSpecified = NotSpecified()
                 ) -> T|str:
    ...


def fill_default(value: T|NotSpecified, *, 
                 env_variable: str|NotSpecified=NotSpecified(), 
                 default_value: T|NotSpecified=NotSpecified(),
                 default_factory: Factory[T]|NotSpecified=NotSpecified(),
                 name: str|None = None, 
                 type_convert: Callable[[str], T]|NotSpecified= NotSpecified(), 
                 ) -> T|str:
    if not isinstance(value, NotSpecified):
        return value
    if not isinstance(env_variable, NotSpecified)\
            and env_variable in os.environ:
        val =  os.environ[env_variable]
        if isinstance(type_convert, NotSpecified):
            converted_val = val
        else: converted_val = type_convert(val)
        return converted_val
    if not isinstance(default_value, NotSpecified):
        return default_value
    if not isinstance(default_factory, NotSpecified):
        return default_factory()

    raise ValueError(f"Couldn't infer default value for {name}, environment variable {env_variable} is not set")
