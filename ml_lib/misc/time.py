"""simple profiling stuff
"""
from typing import Callable, TypeAlias
from functools import wraps
from inspect import signature
from logging import info, getLogger; default_logger=getLogger()
from time import perf_counter

old_time = perf_counter()
function_times: dict[Callable, list[tuple[float, dict]]] = dict()

LogFunction: TypeAlias = Callable[[str], None]


def debug_time(out_string=None, log: LogFunction=info):
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

def time_profile(function, arguments=None, log: LogFunction=info):
    """Makes a function print (to info channel or to the given logging function) the time elapsed during its execution if is_debug is true
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
        log(f"{function.__name__}: time elapsed {delta_t}")
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
