import contextlib
import os
import logging
from ml_lib.misc.data_structures import NotSpecified
from ml_lib.misc.basic import fill_default


@contextlib.contextmanager
def set_num_threads(num_threads: int):
    import torch
    old_n_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(old_n_threads)
    
@contextlib.contextmanager
def set_log_level(level: int, logger: logging.Logger = logging.root):
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
        
@contextlib.contextmanager
def torch_profile_auto_save(profiler, trace_file="trace.json"):
    try:
        with profiler as prof:
            yield
    finally:
        if "prof" in vars():
            prof.export_chrome_trace(trace_file)#type:ignore
        else:
            logging.warning("Could not export stack trace, profiler didn't get constructed correctly, or got destroyed")

@contextlib.contextmanager
def torch_set_sync_debug_mode(debug_mode=1):
    import torch
    import torch.cuda
    old_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode(debug_mode)
        yield
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
