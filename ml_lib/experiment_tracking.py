"""
Experiment tracking / saving (with sqlite)
Still writing, not even nearly production ready (or even working) dont use
"""
from typing import Optional, Any
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey, String, JSON, Column, Integer, Float, Boolean, DateTime, PickleType
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass



class Model(Base):
    __tablename__ = 'models'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(String)

    model_type: Mapped[str] = mapped_column(String)
    parameters: Mapped[Optional[dict]] = mapped_column(JSON)

class Checkpoint(Base):
    __tablename__ = 'checkpoints'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    model_id: Mapped[int] = mapped_column(Integer, ForeignKey('models.id'))
    model: Mapped[Model] = relationship('Model', back_populates='checkpoints')

    step_id: Mapped[int] = mapped_column(Integer, ForeignKey('steps.id'))
    is_last: Mapped[bool] = mapped_column(Boolean)

    checkpoint: Mapped[Any] = mapped_column(PickleType) 

class Training_run(Base):
    __tablename__ = 'training_runs'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    training_parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey('models.id'))
    model: Mapped[Model] = relationship('Model', back_populates='training_runs')

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

class Test(Base):
    __tablename__ = 'tests'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    checkpoint_id: Mapped[int] = mapped_column(Integer, ForeignKey('checkpoints.id'))
    checkpoint: Mapped[Checkpoint] = relationship('Checkpoint', back_populates='tests')

    test_name: Mapped[str] = mapped_column(String)
    test_parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    test_results: Mapped[Any] = mapped_column(PickleType)


class Experiment(Base):
    __tablename__ = 'experiments'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(String)

    models: Mapped[Model] = relationship('Model', secondary='experiment_models', back_populates='experiments')
    tests: Mapped[Test] = relationship('Test', secondary='experiment_tests', back_populates='experiments')
    training_runs: Mapped[Training_run] = relationship('Training_run', secondary='experiment_training_runs', 
                                                       back_populates='experiments')
