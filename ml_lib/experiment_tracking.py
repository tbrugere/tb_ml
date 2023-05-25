"""
Experiment tracking / saving (with sqlite)
Still writing, not even nearly production ready (or even working) dont use
"""
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey, String, JSON, Column, Integer, Float, Boolean, DateTime, PickleType
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Experiment(Base):
    __tablename__ = 'experiments'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(String)
    tags: Mapped[Optional[str]] = mapped_column(String)


class Model(Base):
    __tablename__ = 'models'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(String)
    tags: Mapped[Optional[str]] = mapped_column(String)

    parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    model: Mapped[Optional[PickleType]] = mapped_column(PickleType)
