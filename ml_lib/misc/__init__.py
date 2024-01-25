from typing import Callable, Sequence, Final, Iterable

from contextlib import contextmanager
import functools as ft
from functools import wraps
from inspect import signature
from logging import info
import os
from math import log
from time import perf_counter
import warnings
from pathlib import Path


old_time = perf_counter()
function_times: dict[Callable, list[tuple[float, dict]]] = dict()

#------------------------------------------------------
# simple profiling stuff
#------------------------------------------------------

def debug_time(out_string=None, log=info):
    """only works if is_debug is True
    If out_string is true, prints out_string as well as the time elapsed since the last call to debug_time

    Args:
      out_string: (Default value = None)
      log: what function to use to log (Default value = logging.info)

    Returns:

    """
    global old_time
    if out_string is None:
        old_time = perf_counter()
        return
    new_time = perf_counter()
    log(f"{out_string}: time elapsed {new_time - old_time}s")
    old_time = new_time

def time_profile(function, arguments=None):
    """Makes a function print (to info channel) the time elapsed during its execution if is_debug is true
    warning: the value of is_debug is only checked at the time the decorator is called !

    Args:
      function: 

    Returns:

    """
    if arguments is None:
        arguments = []
    @wraps(function)
    def inner(*args, **kwargs):
        """

        Args:
          *args: 
          **kwargs: 

        Returns:

        """
        global function_times

        bound_args = signature(function).bind(*args, **kwargs)
        bound_args.apply_defaults()
        argument_values = {key: bound_args.arguments[key] for key in arguments}
        t0 = perf_counter()
        ret_val = function(*args, **kwargs)
        delta_t = perf_counter() - t0
        info(f"{function.__name__}: time elapsed {delta_t}")
        function_times[function].append((delta_t, argument_values))
        return ret_val
    
    function_times[function] = []
    return inner

def print_function_times():
    """ """
    global function_times
    for function, times in function_times.items():
        n = len(times)
        mean = sum(time for time, _ in times) / n
        print(f"{function.__name__}: executed {n} times - mean time {mean} seconds")

#------------------------------------------------------------------
# random stuff that should be in the standard lib
#------------------------------------------------------------------

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

