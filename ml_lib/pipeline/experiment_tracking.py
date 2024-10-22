"""
Experiment tracking / saving (with sqlite)
Still writing, not even nearly production ready (or even working) dont use
"""
from time import strftime
from uuid_extensions import uuid7 # TODO: when this is merged into python, remove the dependency
from typing import Iterable, Optional, Any, TYPE_CHECKING, Self, assert_never
from dataclasses import dataclass, field
from datetime import datetime
from textwrap import indent
from sqlalchemy import create_engine, select, text, MetaData
from sqlalchemy import ForeignKey, String, JSON, Column, Integer, Float, Boolean, DateTime, PickleType, Select, Table, text, Uuid
from sqlalchemy.types import JSON, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, object_session, relationship, Session, MappedAsDataclass
from ml_lib.misc.data_structures import Maybe
from ml_lib.misc.torch_functions import move_batch_to, detach_object

if TYPE_CHECKING:
    import torch

from ml_lib.models.base_classes import Model as Model_
from ml_lib.models import register as model_register
from ml_lib.misc import auto_repr


class Base(DeclarativeBase, ):
    pass

metadata_obj : MetaData= MetaData(schema="experiment_tracking_cache")
class CacheBase(DeclarativeBase):
    metadata = metadata_obj 

def create_tables(engine):
    with engine.connect() as connection :
        connection.execute(text("pragma journal_mode=wal;"))
        connection.execute(text("pragma auto_vacuum=incremental;"))
        connection.execute(text("pragma page_size=16384;"))
    Base.metadata.create_all(engine)

experiment_models = Table(
    "experiment_models",
    Base.metadata,
    Column("experiment_id", ForeignKey("experiments.id"), primary_key=True),
    Column("model_id", ForeignKey("models.id"), primary_key=True),
)

experiment_tests = Table(
    "experiment_tests",
    Base.metadata,
    Column("experiment_id", ForeignKey("experiments.id"), primary_key=True),
    Column("test_id", ForeignKey("tests.id"), primary_key=True),
)

# experiment_training_runs = Table(
#     "experiment_training_runs",
#     Base.metadata,
#     Column("experiment_id", ForeignKey("experiments.id"), primary_key=True),
#     Column("training_run_id", ForeignKey("training_runs.id"), primary_key=True),
# )

# experiment_tests = Table(


@auto_repr("id",  "model_type", "name", "description")
class Model(Base):
    __tablename__ = 'models'
    id: Mapped[Uuid] = mapped_column(Uuid, primary_key=True, default=uuid7)

    name: Mapped[Optional[str]] = mapped_column(String, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String)

    model_type: Mapped[str] = mapped_column(String)
    parameters: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    checkpoints: Mapped[list["Checkpoint"]] = relationship('Checkpoint')
    training_runs: Mapped[list["Training_run"]] = relationship('Training_run')
    experiments: Mapped[list["Experiment"]] = relationship('Experiment', secondary=experiment_models)

    @classmethod
    def from_model(cls, model: Model_):
        return cls(
            id=model.id,
            name=model.model_name,
            description=model.description,
            model_type=model.get_model_type(), 
            parameters=model.get_hyperparameters(serializable=True)
        )

    def load_model(self, 
                   load_latest_checkpoint: bool = True, 
                   allow_nonfinal_checkpoint: bool = False, 
                   session: Optional[Session] = None):
        if session is None: 
            session = Session.object_session(self)
        model_type = model_register[self.model_type]
        if self.parameters is None: 
            parameters = {}
        else: 
            parameters = self.parameters
        model = model_type(**parameters, name=self.name, db_session=session)

        if load_latest_checkpoint:
            if session is None: raise ValueError("need a session if load_latest_checkpoint is True, but none provided, and the object is not attached")
            checkpoint = self.latest_checkpoint(session, 
                                                allow_nonfinal_checkpoint=allow_nonfinal_checkpoint)
            if checkpoint is None: 
                raise ValueError("no checkpoints found")
            model.load_checkpoint(checkpoint.checkpoint)

        return model

    def has_finished_training(self, with_checkpoint=True):
        if with_checkpoint:
            return self.latest_checkpoint() is not None
        else:
            raise NotImplementedError("What's the point of checking if the training finished but didn't checkpoint at the end ???")

    def latest_checkpoint(self, session: Session|None = None,*,  allow_nonfinal_checkpoint:bool=False) -> Optional["Checkpoint"]:
        if session is None: session = object_session(self)
        assert session is not None
        return session.execute(self.latest_checkpoint_query(allow_nonfinal_checkpoint)).scalar_one_or_none()

    def latest_checkpoint_query(self, allow_nonfinal_checkpoint: bool=False) -> Select[tuple["Checkpoint"]]:
        query =  select(Checkpoint)\
                .where(Checkpoint.model_id == self.id)
        if not allow_nonfinal_checkpoint:
            query = query.where(Checkpoint.is_last == True)
        query = query\
                .order_by(Checkpoint.id.desc())\
                .limit(1)
        return query

    @classmethod
    def get_by_name(cls, name: str,  session: Session) -> Self|None:
        query = select(cls).where(cls.name == name)
        model_or_none = session.execute(query).one_or_none()
        if model_or_none is None: return None
        model, = model_or_none
        return model

    def get_info_str(self: Self|None=None, name:str|None=None, long=True):
        if self is None:
            return f"✗ — {name}"
        name = name or self.name
        tr_info_strs = [training_run.get_info_str()
                        for training_run in self.training_runs]
        started_training = len(tr_info_strs) != 0
        finished_training = self.has_finished_training()
        match (started_training, finished_training):
            case (_, True): status_icon = "✓"
            case (True, False): status_icon = "…"
            case (False, False): status_icon = "✗"
            case something_else: assert_never(something_else)

        if long:
            tr_info_str = "\n—".join(f"- {i}" for i in tr_info_strs)
            tr_info_str = "\n" + indent(tr_info_str, prefix="   ")
        else: tr_info_str = ""

        return f"{status_icon} — {name}{tr_info_str}"


class Checkpoint(Base):
    __tablename__ = 'checkpoints'
    id: Mapped[Uuid] = mapped_column(Uuid, primary_key=True, default=uuid7)
    is_last: Mapped[bool] = mapped_column(Boolean)
    checkpoint: Mapped[bytes] = mapped_column(LargeBinary) 

    model_id: Mapped[Uuid] = mapped_column(Uuid, ForeignKey('models.id'))
    model: Mapped[Model] = relationship('Model', back_populates='checkpoints')

    step_id: Mapped[Uuid] = mapped_column(Uuid, ForeignKey('steps.id'))
    step: Mapped["Training_step"] = relationship('Training_step', back_populates='checkpoints')

    tests: Mapped[list["Test"]] = relationship('Test', back_populates='checkpoint')

    @classmethod
    def from_model(cls, 
                   model: Model_, 
                   step: Optional["Training_step"] = None, 
                   is_last: bool = False, 
                   *, 
                   session: Session):
        assert session is not None
        model_obj = model.get_database_object(session=session)
        if model_obj is None:
            raise ValueError("Couldn't infer database object for model, so couldn't checkpoint...")
        return cls(
            model=model_obj,
            step=step,
            is_last=is_last,
            checkpoint=model.get_checkpoint()
        )

    @classmethod
    def from_descriptor_string(cls, string: str, session: Session):
        model_name, *eventually_descriptor = string.split(":")
        model = Model.get_by_name(model_name, session)
        match eventually_descriptor:
            case _ if model is None: return None
            case [] | ["latest"]: return model.latest_checkpoint()
            case [str() as max_step_number_str]:
                max_step_number = int(max_step_number_str)
                request = select(cls)\
                        .where(cls.model_id == model.id)\
                        .join(Training_step)\
                        .where(Training_step.step <= max_step_number)\
                        .order_by(Checkpoint.id.desc())\
                        .limit(1)
            case [str() as training_run_number_str, str() as max_step_number_str]:
                training_run_number = int(training_run_number_str)
                max_step_number = int(max_step_number_str)
                training_run = model.training_runs[training_run_number]
                request = select(cls)\
                        .join(Training_step)\
                        .where(Training_step.training_run_id == training_run.id)\
                        .where(Training_step.step <= max_step_number)\
                        .order_by(Checkpoint.id.desc())\
                        .limit(1)
            case _:
                raise ValueError(f"invalid descriptor string {string}")
        maybe_checkpoint = session.execute(request).one_or_none()
        if maybe_checkpoint is None: return maybe_checkpoint
        checkpoint, = maybe_checkpoint
        return checkpoint

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model.name}, step={self.step.step})"

@auto_repr("id", "experiment_id", "model_id", "steps", "model")
class Training_run(Base):
    __tablename__ = 'training_runs'
    id: Mapped[Uuid] = mapped_column(Uuid, primary_key=True, default=uuid7)

    training_parameters: Mapped[Optional[dict]] = mapped_column(JSON)

    model_id: Mapped[Uuid] = mapped_column(Uuid, ForeignKey('models.id'))
    model: Mapped[Model] = relationship('Model', back_populates='training_runs')
    n_steps: Mapped[int] = mapped_column(Integer)
    steps: Mapped[list["Training_step"]] = relationship('Training_step', back_populates='training_run')

    experiment_id: Mapped[Uuid] = mapped_column(Uuid, ForeignKey('experiments.id'))
    experiment: Mapped["Experiment"] = relationship('Experiment', back_populates="training_runs")

    def last_checkpoint(self, max_step_n: int|None=None, session=None) -> Checkpoint|None:
        if session is None:
            session = Session.object_session(self)
        assert session is not None
        query = select(Checkpoint)\
                .join(Checkpoint.step)\
                .where(Training_step.training_run == self)
        if max_step_n is not None:
            query = query.where(Training_step.step <= max_step_n)
        query = query.order_by(Training_step.step.desc()).limit(1)
        result = session.execute(query).one_or_none()
        match result:
            case None: return None
            case (Checkpoint() as c,): return c
            case anything_else: raise ValueError(f"Unexpected query result {anything_else}")

    def is_finished(self, with_checkpoint=True):
        if with_checkpoint:
            return self.last_checkpoint() is not None
        else:
            raise NotImplementedError("What's the point of checking if the training finished but didn't checkpoint at the end ???")

    def time_span(self):
        from sqlalchemy import func
        query = select(func.min(Training_step.step_time), func.max(Training_step.step_time))
        session = Session.object_session(self)
        if session is None: raise ValueError("Object is not in a database session, cannot pull info")
        time_start, time_end = session.execute(query).one()
        return time_start, time_end
            
    def get_info_str(self):
        name_info = f"training run {self.id}"
        time_start, time_end = self.time_span()
        if time_start is None:
            time_info = "NEVER RAN."
        else:
            assert time_end is not None
            time_start = time_start.strftime("%d/%m %H:%M")
            time_end = time_end.trftime("%d/%m %H:%M")
            time_info = f"{time_start} - {time_end}"

        last_checkpoint = self.last_checkpoint()
        if last_checkpoint is None:
            last_checkpoint_n = "NO CHECKPOINT"
        else: last_checkpoint_n = last_checkpoint.step.step

        n_steps= self.n_steps
        
        progress_info = f"{last_checkpoint_n}/{n_steps}"

        match last_checkpoint:
            case None: status_icon = "✗"
            case Checkpoint(is_last=True): status_icon = "✓"
            case Checkpoint(): status_icon = "…"
            case something_else: assert_never(something_else)

        return f"{status_icon} — {name_info} — {progress_info} — {time_info}"


        
    def get_parameter(self, parameter_name: str):
        if self.training_parameters is None:
            return None
        return self.training_parameters.get(parameter_name, None)
        

    def __str__(self):
        session = Session.object_session(self)
        if session is None:
            return f"{self.__class__.__str__}(model={self.model.name}, id={id})"
        last_step_query = select(Training_step)\
                .where(Training_step.training_run_id== self.id)\
                .order_by(Training_step.step.desc())\
                .limit(1)
        last_step = session.execute(last_step_query).one_or_none()
        if last_step is None:
            return f"{self.__class__.__str__}(model={self.model.name})"
        return f"""{self.__class__.__str__}(
                model={self.model.name}, 
                last_step={last_step.step}, last_step_time={last_step.step_time}, 
                last_step_loss={last_step.loss}, 
                last_step_metrics={last_step.metrcs})"""

@auto_repr("id", "training_run_id", "step", "epoch", "loss")
class Training_step(Base):
    __tablename__ = 'steps'
    id: Mapped[Uuid] = mapped_column(Uuid, primary_key=True, default=uuid7)

    training_run_id: Mapped[int] = mapped_column(Uuid, ForeignKey('training_runs.id'))
    training_run: Mapped[Training_run] = relationship('Training_run', back_populates='steps')

    step: Mapped[int] = mapped_column(Integer)
    epoch: Mapped[int] = mapped_column(Integer)
    step_time: Mapped[DateTime] = mapped_column(DateTime)

    loss: Mapped[Optional[float]] = mapped_column(Float)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)

    checkpoints: Mapped[list[Checkpoint]] = relationship(Checkpoint)

    def __str__(self):
        checkpoint = "CHECKPOINTED" if self.checkpoints else ""
        return f"{self.step}: epoch {self.epoch}, loss {self.loss}, {checkpoint}"


class Test(Base):
    __tablename__ = 'tests'
    id: Mapped[Uuid] = mapped_column(Uuid, primary_key=True, default=uuid7)

    checkpoint_id: Mapped[Uuid] = mapped_column(Integer, ForeignKey('checkpoints.id'))
    checkpoint: Mapped[Checkpoint] = relationship('Checkpoint', back_populates='tests')

    test_name: Mapped[str] = mapped_column(String)
    test_parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    test_results: Mapped[Any] = mapped_column(PickleType)

    experiments: Mapped[list["Experiment"]] = relationship('Experiment', secondary=experiment_tests)




@auto_repr("id", "name", "description")
class Experiment(Base):
    __tablename__ = 'experiments'

    id: Mapped[Uuid] = mapped_column(Uuid, primary_key=True, default=uuid7)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String)

    models: Mapped[list[Model]] = relationship('Model', secondary=experiment_models, back_populates='experiments')
    tests: Mapped[Test] = relationship('Test', secondary=experiment_tests, back_populates='experiments')
    training_runs: Mapped[Training_run] = relationship('Training_run',) 
                                                       # secondary='experiment_training_runs', 
                                                       # back_populates='experiments')

"""
**Cache for step information. Avoids synchronously pulling data from GPU too often**
"""

@dataclass
class CacheCheckpoint():
    """Checkpoint in the :class:`NonBlockingStepCache`"""
    checkpoint: bytes 
    model_id: int
    is_last: bool = False

    @classmethod
    def from_model(cls, 
                   model: Model_, 
                   is_last: bool = False, 
                   *, 
                   session: Session|None):
        model_id = model.get_database_id(session=session)
        if model_id is None:
            raise ValueError("couldn't infer model id...")
        return cls(
            model_id=model_id, 
            is_last=is_last,
            checkpoint=model.get_checkpoint()
        )

    def to_database_object(self, step):
        return Checkpoint(
            checkpoint=self.checkpoint, 
            is_last=self.is_last,
            model_id=self.model_id, 
            step=step
                )

@dataclass
class NonBlockingStep():
    """Step information to be cached in the :class:`NonBlockingStepCache`"""
    training_run_id: int = field()
    step: int = field()
    epoch: int = field()
    step_time: datetime = field()
    loss: Any = field() #type: torch.Tensor
    metrics: dict = field(default_factory=dict)
    checkpoint: CacheCheckpoint|None = field(default=None)

    def __post_init__(self):
        if self.loss is not None:
            self.loss = self.loss.detach().to("cpu", non_blocking=True)
        self.metrics = {i: detach_object(v, ignore_failure=True) 
                        for i, v in self.metrics.items()}
        self.metrics = {i: move_batch_to(v, device="cpu", non_blocking=True, ignore_failure=True) 
                        for i, v in self.metrics.items()}

    def to_database_objects(self) -> Iterable[Training_step|Checkpoint] :
        step = Training_step(
                training_run_id=self.training_run_id,
                step = self.step, 
                epoch=self.epoch,
                step_time=self.step_time, 
                loss = self.loss.item() if self.loss is not None else None, 
                metrics = self.itemize_metrics(self.metrics)
                )

        checkpoint = Maybe()
        if self.checkpoint is not None:
            checkpoint = Maybe(self.checkpoint.to_database_object(step))
        return step, *checkpoint
     
    @staticmethod
    def itemize_metrics(metrics: dict):
        d = dict()
        for k, v in metrics.items():
            if hasattr(v, "item"):
                    v = v.item()
            d[k] = v
        return d

class NonBlockingStepCache():
    """
    **Cache for step information. Avoids synchronously pulling data from GPU too often**

    """
    cached_values: list[NonBlockingStep] = []

    def __init__(self) -> None:
        self.cached_values = []

    def __iter__(self):
        for nbs in self.cached_values:
            yield from nbs.to_database_objects()

    def add(self, step: NonBlockingStep):
        self.cached_values.append(step)

    def empty_into_db(self, session: Session, allow_failure: bool=True):
        """
        Tries to empty itself into the database.
        In case of failure, rolls back the changes on DB and keeps the values in cache.
        then 
        if succeded: returns True
        if failed and allow_failure is False, reraises the error
        if failed and allow_failure is False, returns false
        """
        import sqlite3
        from sqlalchemy.exc import OperationalError
        try:
            for item in self:
                session.add(item)
            session.commit()
        except (sqlite3.OperationalError, OperationalError) as exception:
            session.rollback()
            if not allow_failure:
                raise
            return False
        else:
            self.empty_cache()
            return True

    def empty_cache(self):
        import gc
        self.cached_values = []
        gc.collect()

