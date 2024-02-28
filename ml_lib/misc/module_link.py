from types import ModuleType
import importlib.util
from copy import copy
import sys

def module_link(name, targetname):
    module = importlib.import_module(targetname)
    sys.modules[name] = module
    return module

def lazy_import(name) -> ModuleType:
    # directly taken from the importlib examples
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

def lazy_module_link(name, targetname)-> ModuleType:
    """set name to __name__.name"""
    spec = importlib.util.find_spec(targetname)
    spec = copy(spec)
    assert spec is not None and spec.loader is not None
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

