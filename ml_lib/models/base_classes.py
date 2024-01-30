from typing import (Callable, Optional, TypeVar, Annotated, get_type_hints, 
                    Final, Literal, overload, TYPE_CHECKING,
                    Any, ParamSpec, Generic)
# dataclass_transform <-- wait for python3.12
if TYPE_CHECKING:
    from ..experiment_tracking import Model as Database_Model

import functools as ft
import itertools as it
from io import BytesIO, StringIO
from logging import getLogger; log = getLogger(__name__)
from types import FunctionType
from inspect import signature
from pathlib import Path

from sqlalchemy.orm import Session
import torch
import torch.nn as nn

from ..environment import HasEnvironmentMixin
from ..misc import human_readable

Parameters = ParamSpec("Parameters")
LossParameters = ParamSpec("LossParameters")
ReturnType = TypeVar("ReturnType")
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
        for attr_name, attr in class_dict.items():
            if isinstance(attr, FunctionType) and attr_name not in ("__init__", "device"):
                attr = cls.use_model_context(attr)
            new_class_dict[attr_name] = attr
        c = type.__new__(cls, name, bases, new_class_dict)
        cls.validate(c)
        return c
        

    @staticmethod
    def use_model_context(f):
        __tracebackhide__ = True
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

class HasLossMixin(Generic[LossParameters]):
    """
    Mixin for models that have a loss function.
    """

    @staticmethod 
    def no_loss_error(*args: LossParameters.args, **loss_params: LossParameters.kwargs) -> torch.Tensor:
        """Procedure used at training time to compute the loss"""
        del args
        del loss_params
        raise NotImplementedError("Loss function should be defined")
    
    """Procedure used at training time to compute the loss"""
    compute_loss: Callable[LossParameters, torch.Tensor] = no_loss_error



# @dataclass_transform(kw_only_default=True, field_specifiers=(Hyperparameter[Any],))
class Model(nn.Module, HasEnvironmentMixin, HasLossMixin[LossParameters], 
            Generic[Parameters, ReturnType, LossParameters], 
            metaclass=ModelMeta):
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
     
    name: Final[Optional[str]]=None
    model_name: Optional[str]=None
    description: Optional[str]=None
    id: Optional[int]=None # id of the model in the database
    _dummy_param: nn.Parameter

    predict: Optional[Callable] = None
    sample: Optional[Callable] = None
    do_training: Optional[Callable[..., None]] = None
    do_epoch: Optional[Callable[..., None]] = None
    do_pretraining: Optional[Callable[..., None]] = None

    def __init__(self, name: Optional[str]=None, **hyperparameters):
        nn.Module.__init__(self)
        HasEnvironmentMixin.__init__(self)
        # self.device = _get_default_device()
        self.model_name = name
        self._dummy_param = nn.Parameter()
        hyperparameters = self.fill_with_defaults(hyperparameters)
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

    """
    Typing for torch functions: 
    ---------------------------

    Basically tell the typechecker that forward and __call__ have the same 
    signature.
    TODO create a Layer class and set those on that class. They don't make that much sense on the model class, since I often don't even define the forward method
    """

    forward: Callable[Parameters, ReturnType]
    __call__: Callable[Parameters, ReturnType]

    """
    model information
    -----------------
    """

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

    """
    Creating, loading and saving
    ----------------------------
    """

    def save_checkpoint(self, file: Path|str|BytesIO):
        torch.save({
            "model_state_dict": self.state_dict(),
            "hyperparameters": self.get_hyperparameters(), 
            "name": self.model_name,
        }, file)

    def get_checkpoint(self):
        output = BytesIO()
        self.save_checkpoint(output)
        return output.getvalue()

    def load_checkpoint(self, checkpoint:Path|str|BytesIO|bytes|dict):
        if isinstance(checkpoint, bytes):
            checkpoint = BytesIO(checkpoint)
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        assert isinstance(checkpoint, dict)
        self.load_state_dict(checkpoint["model_state_dict"])

    @classmethod
    def from_checkpoint(cls, checkpoint: Path|str|BytesIO):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint)
        assert isinstance(checkpoint, dict)
        model = cls(name=checkpoint.get("name", None), **checkpoint["hyperparameters"]) # type: ignore
        model.load_state_dict(checkpoint["model_state_dict"]) # type: ignore
        return model

    """
    Hyperparameters
    ---------------
    """

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
                    if hasattr(t, "__metadata__") and
                    IS_HYPERPARAM in t.__metadata__]
        return [attr_name for attr_name, t 
                in get_type_hints(cls, include_extras=True).items()
                if hasattr(t, "__metadata__") and
                IS_HYPERPARAM in t.__metadata__]

    def get_hyperparameters(self):
        return {attr_name: getattr(self, attr_name) 
                for attr_name in self.list_hyperparameters()}

    def set_hyperparameters(self, *, allow_missing=False, **hyperparameters):
        required = set(self.list_hyperparameters())
        provided = set(hyperparameters.keys())
        model_name = self.get_model_type()
        match (required-provided, provided-required):
            case [], []:
                pass
            case _, [] if allow_missing:
                pass
            case missing, []:
                raise ValueError(f"Missing hyperparameters for {model_name} : {missing}")
            case [], unknown:
                raise ValueError(f"Unknown hyperparameters for {model_name}: {unknown}")
            case [], []:
                raise ValueError(f"Wrong hyperparameters for {model_name}: \n"
                                 f"expected: {required}\n"
                                 f"provided (including defaults: {provided})\n")

        for attr_name, value in hyperparameters.items():
            setattr(self, attr_name, value)

    @classmethod
    def fill_with_defaults(cls, partial_hyperparameters, allow_missing=False):
        for attr_name in cls.list_hyperparameters(return_types=False):
            if attr_name not in partial_hyperparameters and hasattr(cls, attr_name):
                partial_hyperparameters[attr_name] = getattr(cls, attr_name)
            elif attr_name not in partial_hyperparameters and not allow_missing:
                raise ValueError(f"Missing hyperparameter {attr_name}, and no default value is set")
        return partial_hyperparameters

    def hash_hyperparameters(self):
        """return a (kinda) unique identifier for the set of hyperparameters"""
        return hash(frozenset(self.get_hyperparameters().items()))

    def to_database_object(self):
        from ..experiment_tracking import Model as Database_Model
        return Database_Model.from_model(self)

    def get_database_object(self, session: Session) -> Optional["Database_Model"]:
        from ..experiment_tracking import Model as Database_Model
        if self.id is not None:
            return session.get(Database_Model, self.id)
        elif self.model_name is not None:
            return session.query(Database_Model).filter_by(name=self.model_name).first()
        else:
            return None

    def sync_with_database_object(self, session: Session) -> None:
        database_object = self.get_database_object(session)
        if database_object is not None:
            self.id = database_object.id
            self.model_name = database_object.name

    def save_to_database(self, session: Session, replace:bool = False) -> None:
        from ..experiment_tracking import Model as Database_Model
        if self.id is not None:
            existing = session.get(Database_Model, self.id)
            database_object = self.to_database_object()
            match existing, replace:
                case None, True:
                    session.add(database_object)
                case None, False:
                    log.warn(f"Model {self} is not in the database, but has an id. Adding")
                    session.add(database_object)
                case _, True:
                    session.merge(database_object)
                case _, False:
                    raise ValueError(f"Model {self} is already in the database, and replace is False")
        elif self.model_name is not None:
            existing = session.query(Database_Model).filter_by(name=self.model_name).first()
            if existing is not None:
                self.id = existing.id

            database_object = self.to_database_object()
            match existing, replace:
                case None, True:
                    session.add(database_object)
                case None, False:
                    log.warn(f"Model {self} is not in the database, but has a name. Adding")
                    session.add(database_object)
                case _, True:
                    session.merge(database_object)
                case _, False:
                    raise ValueError(f"Model {self} is already in the database, and replace is False")
        else: 
            database_object = self.to_database_object()
            session.add(self.to_database_object())

        # import pdb; pdb.set_trace()
        # assert database_object in session
        session.commit()
        self.sync_with_database_object(session)




    """
    Capabilities
    ------------
    """

    def has_pretraining_function(self):
        """returns True if the model has a `do_pretraining` function
        This function will be called at the start of the training if available
        """
        return hasattr(self, "do_pretraining")

    def has_training_function(self):
        """returns True if the model has a `do_training` function
        This function will be used instead of the loss to train the model if available
        """
        return self.do_training is not None

    def has_epoch_function(self):
        """returns True if the model has a `do_epoch` function
        This function will be called instead of the loss to train the model if available 
        (and `has_training_function` is False)
        """
        return hasattr(self, "do_epoch")

    @property
    def is_supervised(self) -> bool:
        return self.predict is not None

    @property
    def is_generative(self) -> bool:
        return self.generate is not None

    """
    Misc
    ----
    """

    @classmethod
    def get_model_type(cls):
        if hasattr(cls, "name") and cls.name is not None:
            return cls.name# type: ignore
        return cls.__name__

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

class Generative(Unsupervised):

    def sample(self, n: int = 1) -> torch.Tensor:
        """sample from the model"""
        raise NotImplementedError

class SelfSupervised(Supervised):

    def generate_input(self, x: torch.Tensor) -> torch.Tensor:
        """generate an input for the model using a datapoint"""
        raise NotImplementedError

    def generate_ground_truth(self, x: torch.Tensor) -> torch.Tensor:
        """generate a ground truth for the model using a datapoint"""
        raise NotImplementedError

    def compute_loss(self, x, loss_fun: Optional[Callable]=None):
        x = self.generate_input(x)
        gt = self.generate_ground_truth(x)
        return super().compute_loss(x, gt, loss_fun=loss_fun)

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

    # def __init__(self, body, head):
    #     super().__init__()
    #     self.body = body
    #     self.head = head

    def predict(self, x):
        return self.head(self.body(x))

    def compute_loss(self, x, gt, **kwargs):
        y = self.body(x)
        loss = self.head.compute_loss(y, gt, **kwargs)
        return loss
