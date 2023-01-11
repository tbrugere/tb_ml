from . import CAE # run every submodule in order to register every model

from ..register import Loader
from .registration import register

load_model = Loader(register)


