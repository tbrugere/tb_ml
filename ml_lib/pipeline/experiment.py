from typing import Any, Literal, overload, TYPE_CHECKING

from copy import deepcopy
import functools as ft
from io import TextIOBase
from logging import getLogger; log = getLogger(__name__)
from pathlib import Path

from pydantic import (BaseModel, Field)
import yaml

if TYPE_CHECKING:
    import torch
    from sqlalchemy.orm import Session as DBSession
    from .experiment_tracking import Training_run as DBTraining_run, Experiment as DBExperiment 


from ml_lib.register import Loader
from ml_lib.models import Model, register as model_register
from ml_lib.datasets import register as dataset_register, Dataset, transform_register
from .trainer import Training_parameters

class ModelConfig(BaseModel):
    type: str = Field()
    name: str = Field()
    params: dict[str, Any] = Field(default={})
    training_parameters: Training_parameters = Field(
            default=lambda: Training_parameters(n_epochs=1))
    training_set: str = "train"
    testing_set: str = "test"


    def load_model(self, additional_params = None, register=model_register, 
                   load_checkpoint=False, 
                   database_session: "DBSession|None" = None
                   ):
        additional_params = additional_params or dict()
        model_class: type[Model] = register[self.type]#type:ignore
        model = model_class(**self.params, 
                           eventual_additional_hyperparameters=additional_params, 
                           name=self.name)
        if database_session is not None:
            model.sync_with_database_object(database_session)
        if load_checkpoint:
            raise NotImplementedError
        return model

class DatasetConfig(BaseModel):
    type: str = Field()
    params: dict[str, Any] = Field(default={})
    transforms: list[dict|str]

    def load_dataset(self, additional_params = None,
                     dataset_register=dataset_register, 
                     transform_register=transform_register) -> Dataset:
        additional_params = additional_params or dict()
        dataset_class: type[Dataset] = dataset_register[self.type] #type:ignore
        transform_loader = Loader(transform_register)#type:ignore
        transforms = [transform_loader(t) for t in self.transforms]

        dataset = dataset_class(**self.params, **additional_params)
        for transform in transforms:
            dataset = transform(dataset)
        return dataset

class ExperimentConfig(BaseModel):
    models: list[ModelConfig] = Field()
    datasets: dict[str, DatasetConfig] = Field()

    name: str|None = None
    description: str|None = None
    unsafe: bool = False #disables some checks

    @classmethod
    def load_yaml(cls, config_path):
        return cls.model_validate(yaml.safe_load(config_path.read_text()))

    class Config:
        ignored_types = (ft.cached_property,)
        protected_namespaces = ()

class Experiment():
    config: ExperimentConfig
    name: str|None

    _datasets: dict[str, Dataset]

    model_register = model_register
    dataset_register = dataset_register
    transform_register = transform_register

    database_session : "DBSession|None"
    database_object: "DBExperiment|None" 

    def __init__(self, config, name:str|None=None,  
                 database_session: "DBSession|None" = None):
        self.config = config
        self._datasets = {}
        self.database_session = database_session
        self.name = config.name or name
        self.set_database()

    @classmethod
    def from_yaml(cls, yaml_file: Path|TextIOBase|str, database_session: "DBSession|None" = None, name: str|None = None):
        if isinstance(yaml_file, str):
            yaml_file = Path(yaml_file)
        if isinstance(yaml_file, Path):
            if name is None:
                name = yaml_file.name
            with yaml_file.open("r") as f:
                return cls.from_yaml(f, name=name, database_session=database_session)
        assert isinstance(yaml_file, TextIOBase)
        config_dict = yaml.safe_load(yaml_file)
        config = ExperimentConfig.model_validate(config_dict)
        return cls(config, database_session=database_session, name=name)

    ##############################  loading
    def load_dataset(self, which="train", cache=True):
        if self._datasets is None:
            self._datasets = {}
        if which in self._datasets:
            return self._datasets[which]
        if which == "any":
            if self._datasets:
                return next(iter(self._datasets.values()))
            else: which = next(iter(self.config.datasets))
        dataset_config = self.config.datasets[which]
        dataset: Dataset = dataset_config.load_dataset(
                dataset_register=self.dataset_register, 
                transform_register=self.transform_register)

        if cache:
            self._datasets[which] = dataset
        return dataset

    @overload
    def load_model(self, model: int|str, *, return_config: Literal[False] = False, load_checkpoint=False) -> Model:
        ...
    @overload
    def load_model(self, model: int|str, *, return_config: Literal[True], load_checkpoint=False) -> tuple[Model, ModelConfig]:
        ...
    def load_model(self, model: int|str, *,  
                   return_config=False, 
                   load_checkpoint=False,) -> tuple[Model, ModelConfig] | Model:
        model_num: int
        if isinstance(model, int):
            model_num = model
        else: 
            assert isinstance(model, str)
            try:
                model_num = [m.name for m in self.config.models].index(model)
            except ValueError as e:
                raise ValueError(f"{model} was not in the current experiment config") from e
        model_config: ModelConfig = self.config.models[model_num]
        possible_datasets = [model_config.training_set, model_config.testing_set]
        if (already_loaded:= [p for p in possible_datasets if (p in self._datasets)]):
            which_dataset = already_loaded[0]
        else: which_dataset = model_config.training_set
        dataset = self.load_dataset(which_dataset)

        dataset_parameters = dataset.dataset_parameters()

        m: Model = model_config.load_model(
                    additional_params=dataset_parameters, 
                    register=self.model_register, 
                    load_checkpoint=load_checkpoint, 
                    database_session=self.database_session
                    ) #type:ignore
        self.ensure_relationship_to_model(m)
        if return_config:
            return  m, model_config
        else:
            return m

    @property
    def n_models(self):
        return len( self.config.models )
    ####################################################################
    ################ database stuff ####################################

    def set_database(self):
        from .experiment_tracking import Experiment as DBExperiment
        session = self.database_session
        if session is None:
            log.warn("No database session set")
            return
        name = self.name
        if name is None :
            error_str = (
                "Database object given to experiment, but no experiment name"
                     ", so no way to link experiment to database.")
            if not self.config.unsafe:
                raise ValueError(error_str)
            else: 
                log.warn(error_str)
                return
        db_object: DBExperiment|None = session.query(DBExperiment).filter_by(name=name).one_or_none()
        if db_object is not None:
            self.database_object = db_object
            return 
        
        db_object = DBExperiment(
                name=name, 
                description=self.config.description, 
                )
        session.add(db_object)
        session.commit()
        self.database_object = db_object
        return

    def get_training_run_from_db(self, model: Model, resume_from) -> "DBTraining_run| None":
        from sqlalchemy import select, desc
        from ml_lib.pipeline.experiment_tracking import Experiment as DBExperiment, Training_run as DBTraining_run, Training_step as DBTraining_step
        session = self.database_session
        assert session is not None
        db_object = self.database_object
        training_run_query = select(DBTraining_run)\
                .where(DBTraining_run.experiment == db_object)\
                .where(DBTraining_run.model_id == model.id)

        match resume_from:
            case "highest_step":
                training_run_query = training_run_query\
                        .join(DBTraining_run.steps)\
                        .order_by(DBTraining_step.step.desc())\
                        .limit(1)
                result = session.execute(training_run_query).one_or_none()
                if result is None: return None 
                else: return result[0]
            case "highest_time":
                training_run_query = training_run_query\
                        .join(DBTraining_run.steps)\
                        .order_by(DBTraining_step.step_time.desc())\
                        .limit(1)
                result = session.execute(training_run_query).one_or_none()
                if result is None: return None 
                else: return result[0]
            case "only_one":
                result = session.execute(training_run_query).one_or_none()
                if result is None: return None 
                else: return result[0]
            case "ask":
                possible_trainers = session.execute(training_run_query).all()
                if len(possible_trainers) == 0: return None
                if len(possible_trainers) == 1: return possible_trainers[0][0]
                for i, t in enumerate(possible_trainers):
                    print(f"{i}: {str(t)}")
                trainer_index = int(input(f"Which run should we resume [0 - {len(possible_trainers) -1}]"))
                return possible_trainers[trainer_index][0]
            case _:
                raise ValueError(f"Unknown value for resume_from: {resume_from}")
   
    def ensure_relationship_to_model(self, model: Model):
        if self.database_session is None: return
        db_object = self.database_object
        assert db_object is not None
        model_db = model.get_database_object(self.database_session, add_if_needed=True)
        if model_db in db_object.models:
           return 
        assert model_db is not None
        db_object.models.append(model_db)
        self.database_session.commit()
    ####################################################################
    ################ Actually doing stuff ##############################

    def train(self, model: int|str, device:"torch.device|str" = "cpu", 
              resume_from = "highest_step"):
        from ml_lib.pipeline.trainer import Trainer
        import torch
        model_, model_params= self.load_model(model, 
                                              return_config=True, 
                                              load_checkpoint=False)
        data = self.load_dataset(model_params.training_set)

        if resume_from and self.database_object is not None:
            trainer_db = self.get_training_run_from_db(model_, 
                                                       resume_from=resume_from)
        elif resume_from and self.database_object is not None: 
            log.warn("resume_from is set, but no database was given")
            trainer_db = None
        else:
            trainer_db = None

        trainer = Trainer(
                training_parameters=model_params.training_parameters, 
                model = model_, 
                data = data, 
                device=torch.device(device), 
                database=self.database_session, 
                db_experiment=self.database_object, 
                resume_from=trainer_db, 
                )
        trainer.train()

        return model_

    def train_all(self, device:"torch.device|str" = "cpu"):
        for i in range(self.n_models):
            self.train(i, device)

