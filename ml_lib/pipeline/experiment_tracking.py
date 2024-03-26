"""
Experiment tracking / saving (with sqlite)
Still writing, not even nearly production ready (or even working) dont use
"""
from typing import Optional, Any, TYPE_CHECKING, Self
from sqlalchemy import create_engine, select
from sqlalchemy import ForeignKey, String, JSON, Column, Integer, Float, Boolean, DateTime, PickleType, Select, Table
from sqlalchemy.types import JSON, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

from ml_lib.models.base_classes import Model as Model_
from ml_lib.models import register as model_register
from ml_lib.misc import auto_repr

class Base(DeclarativeBase):
    pass

def create_tables(engine):
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
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

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

    def load_model(self, load_latest_checkpoint: bool = True, session: Optional[Session] = None):
        model_type = model_register[self.model_type]
        if self.parameters is None: 
            parameters = {}
        else: 
            parameters = self.parameters
        model = model_type(**parameters)

        if load_latest_checkpoint:
            if session is None: 
                session = Session.object_session(self)
                if session is None: raise ValueError("need a session if load_latest_checkpoint is True, but none provided, and the object is not attached")
            checkpoint = self.latest_checkpoint(session)
            if checkpoint is None: 
                raise ValueError("no checkpoints found")
            model.load_checkpoint(checkpoint.checkpoint)

        return model

    def latest_checkpoint(self, session: Session) -> Optional["Checkpoint"]:
        return session.execute(self.latest_checkpoint_query()).scalar_one_or_none()

    def latest_checkpoint_query(self) -> Select[tuple["Checkpoint"]]:
        return select(Checkpoint)\
                .where(Checkpoint.model_id == self.id)\
                .where(Checkpoint.is_last == True)\
                .order_by(Checkpoint.id.desc())\
                .limit(1)


    @classmethod
    def get_by_name(cls, name: str,  session: Session) -> Self|None:
        query = select(cls).where(cls.name == name)
        model_or_none = session.execute(query).one_or_none()
        if model_or_none is None: return None
        model, = model_or_none
        return model


class Checkpoint(Base):
    __tablename__ = 'checkpoints'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    is_last: Mapped[bool] = mapped_column(Boolean)
    checkpoint: Mapped[bytes] = mapped_column(LargeBinary) 

    model_id: Mapped[int] = mapped_column(Integer, ForeignKey('models.id'))
    model: Mapped[Model] = relationship('Model', back_populates='checkpoints')

    step_id: Mapped[int] = mapped_column(Integer, ForeignKey('steps.id'))
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

@auto_repr("id", "experiment_id", "model_id", "steps", "model")
class Training_run(Base):
    __tablename__ = 'training_runs'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    training_parameters: Mapped[Optional[dict]] = mapped_column(JSON)

    model_id: Mapped[int] = mapped_column(Integer, ForeignKey('models.id'))
    model: Mapped[Model] = relationship('Model', back_populates='training_runs')
    steps: Mapped[list["Training_step"]] = relationship('Training_step', back_populates='training_run')

    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey('experiments.id'))
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
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    training_run_id: Mapped[int] = mapped_column(Integer, ForeignKey('training_runs.id'))
    training_run: Mapped[Training_run] = relationship('Training_run', back_populates='steps')

    step: Mapped[int] = mapped_column(Integer)
    epoch: Mapped[int] = mapped_column(Integer)
    step_time: Mapped[DateTime] = mapped_column(DateTime)

    loss: Mapped[Optional[float]] = mapped_column(Float)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)

    checkpoints: Mapped[Checkpoint] = relationship('Checkpoint', back_populates='step')

    def __str__(self):
        checkpoint = "CHECKPOINTED" if self.checkpoints else ""
        return f"{self.step}: epoch {self.epoch}, loss {self.loss}, {checkpoint}"


class Test(Base):
    __tablename__ = 'tests'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    checkpoint_id: Mapped[int] = mapped_column(Integer, ForeignKey('checkpoints.id'))
    checkpoint: Mapped[Checkpoint] = relationship('Checkpoint', back_populates='tests')

    test_name: Mapped[str] = mapped_column(String)
    test_parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    test_results: Mapped[Any] = mapped_column(PickleType)

    experiments: Mapped[list["Experiment"]] = relationship('Experiment', secondary=experiment_tests)




@auto_repr("id", "name", "description")
class Experiment(Base):
    __tablename__ = 'experiments'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String)

    models: Mapped[list[Model]] = relationship('Model', secondary=experiment_models, back_populates='experiments')
    tests: Mapped[Test] = relationship('Test', secondary=experiment_tests, back_populates='experiments')
    training_runs: Mapped[Training_run] = relationship('Training_run',) 
                                                       # secondary='experiment_training_runs', 
                                                       # back_populates='experiments')
