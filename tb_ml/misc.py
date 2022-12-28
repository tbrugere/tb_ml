from typing import Callable

from functools import wraps
from inspect import signature
from logging import info
from time import perf_counter

from torch import nn


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


old_time = perf_counter()
function_times: dict[Callable, list[tuple[float, dict]]] = dict()

def debug_time(out_string=None):
    """only works if is_debug is True
    If out_string is true, prints out_string as well as the time elapsed since the last call to debug_time

    Args:
      out_string: (Default value = None)

    Returns:

    """
    global old_time
    if out_string is None:
        old_time = perf_counter()
        return
    new_time = perf_counter()
    info(f"{out_string}: time elapsed {new_time - old_time}s")
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


#-----------------------------------------------------------
# Pytorch stuff
#-----------------------------------------------------------

def freeze_model(m: nn.Module):
    for param in m.parameters():
        param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
