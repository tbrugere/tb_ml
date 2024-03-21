from typing import (Callable, Optional, TypeVar, Annotated, get_type_hints, 
                    Final, Literal, overload, TYPE_CHECKING,
                    TypeAlias, 
                    Any, ParamSpec, Generic)
# dataclass_transform <-- wait for python3.12
if TYPE_CHECKING:
    from ml_lib.pipeline.experiment_tracking import Model as Database_Model

import functools as ft
from copy import deepcopy
import itertools as it
from io import BytesIO, StringIO
from logging import getLogger; log = getLogger(__name__)
from types import FunctionType
from inspect import signature
from pathlib import Path

from sqlalchemy.orm import Session
import torch
import torch.nn as nn

from ml_lib.register import LoadableMixin, try_serializing
from ..environment import HasEnvironmentMixin
from ml_lib.misc.typing import get_type_origin, advanced_type_check
from ..misc import human_readable

Parameters = ParamSpec("Parameters")
LossParameters = ParamSpec("LossParameters")
ReturnType = TypeVar("ReturnType")
T = TypeVar("T")
IS_HYPERPARAM = "hyperparameter"
DONT_CHECK = "dontcheck"
Hyperparameter = Annotated[T, IS_HYPERPARAM]
DontCheck = Annotated[T, DONT_CHECK]


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
            ModelMeta.check_argument_devices(args, kwargs, model=self, function=f)
            with device:
                return f(self, *args, **kwargs)
        return wrapped

    @staticmethod
    def check_argument_devices(args, kwargs, * , model, function):
        """Check that the arguments passed to a model functions are on the 
        model's device"""
        if not __debug__:
            return
        sig = signature(function)
        params = sig.parameters
        bound_args = sig.bind_partial(*args, **kwargs)
        for arg_name, value in bound_args.arguments.items():
            param = params[arg_name]
            if not hasattr(value, "device"):
                continue
            if hasattr(param.annotation, "__metadata__") and DONT_CHECK in param.annotation.metadata:
                continue
            if value.device == model.device:
                continue
            raise ValueError(f"passed parameter {arg_name} on device {value.device}, but the model {model.get_model_type()} is on device {model.device}. If this is a false positive, annotate the argument with DontCheck")


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

    def __init__(self, name: Optional[str]=None, 
                 eventual_additional_hyperparameters: dict|None=None, 
                 **hyperparameters):
        nn.Module.__init__(self)
        HasEnvironmentMixin.__init__(self)
        # self.device = _get_default_device()
        self.model_name = name
        self._dummy_param = nn.Parameter()
        if eventual_additional_hyperparameters is None:
            eventual_additional_hyperparameters = {}
        hyperparameters = self.fill_with_defaults(hyperparameters, eventual_additional_hyperparameters)
        self.set_hyperparameters(**hyperparameters, )
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

    def num_parameters_tree(self, depth=3, human_readable: bool=True, skip_empty=False):
        from ml_lib.misc.torch_functions import num_parameters_tree
        return num_parameters_tree(self, depth=depth, 
                                   human_readable=human_readable, 
                                   skip_empty=True)


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

    def get_hyperparameters(self, serializable=False):
        if serializable:
            return {attr_name: try_serializing(getattr(self, attr_name))
                    for attr_name in self.list_hyperparameters()}

        return {attr_name: getattr(self, attr_name) 
                for attr_name in self.list_hyperparameters()}


    def set_hyperparameters(self, *, allow_missing=False, deserialize=True, 
                            **hyperparameters):
        """``eventual_additional_hyperparameters`` contains parameters that may be used to 
        set values, but aren't counted as unknown even if they're not used.
        They are default values not explicitely given, that can be inferred from eg. the dataset.

        priority:
            1. provided
            2. model defaults 
            3. eventual_additional_hyperparameters ("inferred")
        """
        from ml_lib.misc.matchers import EmptySet
        all_hyperparameters_with_types = dict(self.list_hyperparameters(return_types=True))
        required = set(self.list_hyperparameters())
        provided = set(hyperparameters.keys())
        model_name = self.get_model_type()
        match (required-provided, provided-required):
            case EmptySet(), EmptySet():
                pass
            case _, EmptySet() if allow_missing:
                pass
            case missing, EmptySet():
                raise ValueError(f"Missing hyperparameters for {model_name} : {missing}")
            case EmptySet(), unknown:
                raise ValueError(f"Unknown hyperparameters for {model_name}: {unknown}")
            case _, _:
                raise ValueError(f"Wrong hyperparameters for {model_name}: \n"
                                 f"expected: {required}\n"
                                 f"provided (including defaults and inferred: {provided})"                                 )

        for attr_name, value in hyperparameters.items():
            self.set_hyperparameter(attr_name, value, deserialize=deserialize, 
                                    hyperparameters=all_hyperparameters_with_types)

    def set_hyperparameter(self, attr_name, value, 
                           deserialize=True, 
                           hyperparameters: dict[str, type]|None=None):
        if hyperparameters is None:
            hyperparameters = dict(self.list_hyperparameters(return_types=True))

        parameter_type = hyperparameters[attr_name]
        if not advanced_type_check(value, parameter_type):
            try:
                should_deserialize = deserialize and issubclass(get_type_origin(parameter_type), LoadableMixin) and isinstance(value, dict)
            except TypeError:
                should_deserialize = False
            if should_deserialize:
                value = parameter_type.from_config(value)
            else:
                log.warn(f"Setting parameter {attr_name} to value {value}, which is of the wrong type: expected {parameter_type}")
        setattr(self, attr_name, value)
        

    @classmethod
    def fill_with_defaults(cls, partial_hyperparameters, 
                           eventual_additional_hyperparameters, 
                           allow_missing=False):
        for attr_name in cls.list_hyperparameters(return_types=False):
            if attr_name not in partial_hyperparameters and hasattr(cls, attr_name):
                partial_hyperparameters[attr_name] = deepcopy(getattr(cls, attr_name))
            elif attr_name not in partial_hyperparameters and attr_name in eventual_additional_hyperparameters:
                partial_hyperparameters[attr_name] = deepcopy(eventual_additional_hyperparameters[attr_name])
            elif attr_name not in partial_hyperparameters and not allow_missing:
                raise ValueError(f"Missing hyperparameter {attr_name}, and no default value is set")
        return partial_hyperparameters

    def hash_hyperparameters(self):
        """return a (kinda) unique identifier for the set of hyperparameters"""
        return hash(frozenset(self.get_hyperparameters().items()))

    def to_database_object(self):
        from ml_lib.pipeline.experiment_tracking import Model as Database_Model
        return Database_Model.from_model(self)

    def get_database_object(self, session: Session, add_if_needed=False) -> Optional["Database_Model"]:
        from ml_lib.pipeline.experiment_tracking import Model as Database_Model
        if self.id is not None:
            db_object = session.get(Database_Model, self.id)
        elif self.model_name is not None:
            db_object = session.query(Database_Model).filter_by(name=self.model_name).first()
        else:
            return None
        if db_object is not None:
            return db_object
        if add_if_needed:
            self.save_to_database(session)
            return self.get_database_object(session, add_if_needed=False)
        return None

    def sync_with_database_object(self, session: Session) -> None:
        database_object = self.get_database_object(session)
        if database_object is not None:
            self.id = database_object.id
            self.model_name = database_object.name

    def save_to_database(self, session: Session, replace:bool = False) -> None:
        from ml_lib.pipeline.experiment_tracking import Model as Database_Model
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
        return self.do_pretraining is not None

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
        return self.do_epoch is not None

    @property
    def is_supervised(self) -> bool:
        return self.predict is not None

    @property
    def is_generative(self) -> bool:
        return self.sample is not None

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
