from typing import Callable, Optional, TypeVar, Annotated, get_type_hints, Final, Literal, overload

import functools as ft
import itertools as it
from types import FunctionType
from io import BytesIO, StringIO
from inspect import signature
from pathlib import Path

import torch
import torch.nn as nn

from ..environment import HasEnvironmentMixin
from ..misc import human_readable

T = TypeVar("T")
IS_HYPERPARAM = "hyperparameter"
Hyperparameter = Annotated[T, IS_HYPERPARAM]

def _get_default_device():
    return torch.Tensor().device

class ModelMeta(type):
    """
    Metaclass for models. 
    Ensures that all the model’s function runs in the model’s context manager.
    """
    def __new__(cls, name, bases, class_dict):
        new_class_dict = {}
        cls.validate(new_class_dict)
        for attr_name, attr in class_dict.items():
            if isinstance(attr, FunctionType) and attr_name not in ("__init__", "device"):
                attr = cls.use_model_context(attr)
            new_class_dict[attr_name] = attr
        return type.__new__(cls, name, bases, new_class_dict)

    @staticmethod
    def use_model_context(f):
        @ft.wraps(f)
        def wrapped(self, *args, **kwargs):
            device = self.device
            with device:
                return f(self, *args, **kwargs)
        return wrapped

    @staticmethod
    def validate(model_class):
        """
        Different checks for the model class, add more if needed.
        For now check that there is no hyperparameter called
        - device
        - allow_missing
        """
        reserved_names: Final[set[str]] = {"device", "allow_missing"}
        for hyperparameter in model_class.list_hyperparameters():
            if hyperparameter in reserved_names:
                raise ValueError(f"Hyperparameter {hyperparameter} is reserved")

    # @staticmethod
    # def save_parameters_init(__init__):
    #     @ft.wraps(__init__)
    #     def wrapped(self, *args, **kwargs):

class HasLossMixin:
    """
    Mixin for models that have a loss function.
    """
    def compute_loss(self, *args, **loss_params) -> torch.Tensor:
        """Procedure used at training time to compute the loss"""
        del args
        del loss_params
        raise NotImplementedError("Loss function should be defined")


class Model(nn.Module, HasEnvironmentMixin, HasLossMixin, metaclass=ModelMeta):
    """
    Base class for this library’s models.
    Got a bunch of sugar including:
        - keeping track of the device the model is on
        - ensuring that all the model’s function run on the correct device by default
        - somewhere to place loss functions
        - Gestion of hyperparameters
        - saving / loading
        - some useful functions
    """
     

    _dummy_param: nn.Parameter

    def __init__(self, **hyperparameters):
        nn.Module.__init__(self)
        HasEnvironmentMixin.__init__(self)
        # self.device = _get_default_device()
        self._dummy_param = nn.Parameter()
        self.set_hyperparameters(**hyperparameters)
        self.__setup__()

    def __setup__(self):
        """
        This function is called at the end of the constructor.
        It should be used to initialize the model’s parameters.
        At this point, the model's hyperparameters are set, 
        only those should be read in this function.
        """
        pass

    def predict(self, x) -> torch.Tensor: 
        """Procedure used at inference time to compute the output"""
        raise NotImplementedError

    def forward(self, x) -> torch.Tensor: 
        return self.predict(x)

    @property
    def device(self):
        return self._dummy_param.device

    def num_parameters(self, trainable_only: bool = False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def num_parameters_str(self, trainable_only: bool = False, precision: int = 2):
        return human_readable(self.num_parameters(trainable_only), 
                              precision=precision)

    def save_checkpoint(self, file: Path|str|BytesIO):
        torch.save({
            "model_state_dict": self.state_dict(),
            "hyperparameters": self.get_hyperparameters()
        }, file)


    def load_checkpoint(self, checkpoint:Path|str|BytesIO|dict):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint)
        assert isinstance(checkpoint, dict)
        self.load_state_dict(checkpoint["model_state_dict"])

    @classmethod
    def from_checkpoint(cls, checkpoint: Path|str|BytesIO):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint)
        assert isinstance(checkpoint, dict)
        model = cls(**checkpoint["hyperparameters"]) # type: ignore
        model.load_state_dict(checkpoint["model_state_dict"]) # type: ignore
        return model

    @overload
    @classmethod
    def list_hyperparameters(cls, return_types: Literal[False] = False) -> list[str]:
        ...

    @overload
    @classmethod
    def list_hyperparameters(cls, return_types: Literal[True]) -> list[tuple[str, type]]:
        ...

    @classmethod
    def list_hyperparameters(cls, return_types: bool = False):
        if return_types:
            return [(attr_name, t) 
                    for attr_name, t 
                    in get_type_hints(cls, include_extras=True).items()
                    if IS_HYPERPARAM in t.__metadata__]
        return [attr_name for attr_name, t 
                in get_type_hints(cls, include_extras=True).items()
                if IS_HYPERPARAM in t.__metadata__]

    def get_hyperparameters(self):
        return {attr_name: getattr(self, attr_name) 
                for attr_name in self.list_hyperparameters()}

    def set_hyperparameters(self, *, allow_missing=False, **hyperparameters):
        if not allow_missing and set(self.list_hyperparameters()) != set(hyperparameters.keys()):
            raise ValueError(f"Missing hyperparameters: {set(self.list_hyperparameters()) - set(hyperparameters.keys())}")
        elif allow_missing and not set(hyperparameters.keys()).issubset(set(self.list_hyperparameters())):
            raise ValueError(f"Unknown hyperparameters: {set(hyperparameters.keys()) - set(self.list_hyperparameters())}")
        for attr_name, value in hyperparameters.items():
            setattr(self, attr_name, value)

    def __repr__(self):
        s = StringIO()
        s.write(f"{self.__class__.__name__}(\n")
        for attr_name, value in self.get_hyperparameters().items():
            s.write(f"    {attr_name}={value},\n")
        s.write(")")
        return s.getvalue()


class Supervised(Model):

    loss_fun: Optional[Callable]
    """
    Default loss function.
    It should take two arguments, `input` and `target`
    """

    def predict(self, x) -> torch.Tensor: 
        """
        This is the function you should override to implement your model.
        """
        x;
        raise NotImplementedError

    def forward(self, x) -> torch.Tensor: 
        y = self.predict(x)
        return y

    def compute_loss(self, x, gt, loss_fun: Optional[Callable]=None):
        y = self(x)
        if loss_fun is None:
            loss_fun = self.loss_fun
        assert loss_fun is not None, "no loss function "
        loss = loss_fun(y, gt)
        return loss


class Unsupervised(Model):
    pass


class AutoEncoder(Unsupervised):
    """Base class for autoencoders"""

    loss_fun: Optional[Callable]
    encoder: Callable
    decoder: Callable
    """
    Default loss function.
    It should take two arguments, `input` and `target`
    """

    def encode(self, x) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)

    def recognition_loss(self, x, z, loss_fun: Optional[Callable]=None):
        if loss_fun is None: loss_fun = self.loss_fun
        assert loss_fun is not None, "no loss function "
        y = self.decode(z)
        return loss_fun(y, x)

    def predict(self, x):
        return self.encode(x)

    def compute_loss(self, x, loss_fun: Optional[Callable]=None):
        z = self.encode(x)
        loss = self.recognition_loss(x, z, loss_fun)
        return loss

class Head(nn.Module, HasLossMixin, HasEnvironmentMixin):
    """Head modules. These modules are used to change output types"""

    def forward(self, x):
        raise NotImplementedError


class BodyHeadSupervised(Supervised):
    """Base class for models with a body and a head.
    This is a pattern I use often which is why I’m defining it here

    """

    body: nn.Module
    head: Head

    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head

    def predict(self, x):
        return self.head(self.body(x))

    def compute_loss(self, x, gt, **kwargs):
        y = self.body(x)
        loss = self.head.compute_loss(y, gt, **kwargs)
        return loss
