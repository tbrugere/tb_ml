from collections.abc import Sequence
from collections import namedtuple
from typing import Type, Any, Union, Optional, Callable, TYPE_CHECKING
from pydantic import BaseModel, Field

from logging import info
if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DBSession
    from ..experiment_tracking import Training_run as DBTraining_run, Experiment as DBExperiment

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from ..models import load_model
from ..models.base_classes import Model
from ..datasets import load_dataset
from .training_hooks import TrainingHook, OptimizerHook, LRSchedulerHook, DatabaseHook
from ..environment import Environment, HierarchicEnvironment
from ..register import Loader
from .registers import loss_register, training_hook_register

class Training_parameters(BaseModel):
    n_epochs: int = 1
    optimizer: str = "Adam"
    optimizer_arguments: dict =  Field(default_factory=dict)
    lr_scheduler: str|None = None
    loss: dict|None = None # to be gotten from a loss register

    fake_batch_size: int = 1

    clip_grad_norm: float|None = 1.

    step_hooks: list[dict] = []
    epoch_hooks: list[dict] = []
    end_hooks: list[dict] = []

    environment_variables: dict[str, Any] = Field(default_factory=dict)

    performance_tricks: bool = True
    """enables various optimizations. Set to false to help debugging"""

class Trainer():


    model: Model
    data: DataLoader

    training_parameters: Training_parameters
    
    """The eventual loss function"""
    loss: Optional[Callable]

    """The total number of iterations to run (ie n_epochs * len(data))"""
    total_iter: int
    n_epochs: int

    """How many minibatches to run for one optim step"""
    fake_batch_size: int

    """The number of the current iteration"""
    iteration_n: int = 0
    epoch_n: int = 0

    global_env: Environment
    epoch_env: Environment
    iter_env: Environment

    id: int|None = None
    """id in the database"""
    database_object: "DBTraining_run|None" = None

    database_session: "DBSession|None"

    def __init__(self, model: Model, 
                 data: Union[Dataset, DataLoader, Sequence],
                 training_parameters: Training_parameters, 
                 device = "cuda:0", 
                 step_hooks: list[TrainingHook] = [], 
                 epoch_hooks: list[TrainingHook] = [],
                 end_hooks: list[TrainingHook] = [],
                 environment_variables: dict = {}, 
                 database: "DBSession"|None = None, 
                 db_experiment: "DBExperiment|int|None" = None, 
                 resume_from: "int|DBTraining_run|None" = None, 
                 ):
        #TODO: take a train and validation set, or do the separation in-house

        device = torch.device(device)
        self.training_parameters = training_parameters

        if training_parameters.lr_scheduler is not None:
            raise NotImplemented
        match data:
            case DataLoader():
                pass
            case Dataset():
                raise NotImplementedError
        if not isinstance(data, DataLoader):
            raise NotImplemented #TODO wrap it in a DataLoader

        ################ Setup Environments
        self.global_env = Environment()
        self.global_env.record_dict(environment_variables)
        self.epoch_env = HierarchicEnvironment(parent=self.global_env)
        self.iter_env = HierarchicEnvironment(parent=self.epoch_env)

        ################ Actual training stuff
        self.model = model.to(device)
        if training_parameters.loss is None:
            self.loss = None
        else:
            loss_loader = Loader(loss_register)
            self.loss = loss_loader(training_parameters.loss)
        self.data = data
        self.total_iter = len(data) * training_parameters.n_epochs
        self.n_epochs = training_parameters.n_epochs
        self.fake_batch_size = training_parameters.fake_batch_size
        self.device = device
        optimizer_hook = self.get_optimizer_hook(training_parameters.optimizer, 
                                                 training_parameters.optimizer_arguments, 
                                                 clip_grad_norm=training_parameters.clip_grad_norm, 
                                                 fake_batch_size=training_parameters.fake_batch_size)

        ############### Database stuff
        self.database_session=database
        if database is not None:
            assert db_experiment is not None
            from ..experiment_tracking import Experiment as DBExperiment
            if isinstance(db_experiment, int):
                db_experiment=database.get(DBExperiment, db_experiment)
            assert isinstance(db_experiment, DBExperiment)
            # This is where the resume stuff should be 
            self.set_database(database_session=database, experiment=db_experiment, resume_from=resume_from)
            database_hook = self.get_database_hook()


        ################ Record stuff in environment
        if loss is not None:
            self.global_env.record("loss_fun", loss)
        self.global_env.record_dict(dict(
            model=model,
            total_iter = self.total_iter,
            device=device, 
            data=self.data
        ))
        self.global_env.record_dict(training_parameters.model_dump())
        if database is not None:
            self.global_env.record("training_run_id", self.id)
            self.global_env.record("training_run_db", self.database_object)

        ################ set hooks
        hook_loader = Loader(training_hook_register)
        step_hooks = [hook_loader(hook_config) for hook_config in training_parameters.step_hooks] + step_hooks
        epoch_hooks = [hook_loader(hook_config) for hook_config in training_parameters.epoch_hooks] + epoch_hooks
        end_hooks = [hook_loader(hook_config) for hook_config in training_parameters.end_hooks] + end_hooks
        step_hooks.append(optimizer_hook)
        #TODO: add lr_scheduler
        for hook in step_hooks:
            hook.set_environment(self.iter_env)
            hook.setup()
        for hook in epoch_hooks:
            hook.env.set_environment(self.epoch_env)
            hook.setup()
        for hook in end_hooks:
            hook.set_environment(self.global_env)
            hook.setup()
        self.step_hooks = step_hooks
        self.epoch_hooks = epoch_hooks
        self.end_hooks = end_hooks


    def step(self, batch):
        self.iter_env.reset()
        self.iter_env.record_dict(dict(
            iteration=self.iteration_n
        ))
        self.model.train()

        batch = self.move_batch_to(batch, self.device, 
                    non_blocking=self.training_parameters.performance_tricks)
        match batch:
            case dict():
                self.iter_env.record_dict(batch)
            case batch if hasattr(batch, "_asdict"):
                self.iter_env.record_dict(batch._asdict())
            case batch if hasattr(batch, "asdict"):
                self.iter_env.record_dict(batch.asdict())
            case (x, gt):
                self.iter_env.record("x", x)
                self.iter_env.record("gt", gt)
            case _:
                self.iter_env.record("x", batch)
        self.iter_env.record("batch", batch)

        loss = self.iter_env.run_function(self.model.compute_loss, 
                                          record_result_as="loss")
        loss.backward()
        
        for hook in self.step_hooks:
            hook()

    def epoch(self):
        self.epoch_env.reset()
        self.epoch_env.record_dict(dict(
            last_iteration=self.iteration_n, 
            epoch= self.epoch_n
        ))
        for batch in self.data:
            self.step(batch)
            self.iteration_n += 1

        for hook in self.epoch_hooks:
            hook()

    def train(self):
        model = self.model
        if model.do_pretraining is not None:
            info(f"Model {model} has do_pretraining method, launching")
            assert callable(model.do_pretraining), "model do_pretraining is not callable!"
            self.global_env.run_function(model.do_pretraining)

        if model.do_training is not None:
            info(f"Model {model} has do_training method, using that")
            assert callable(model.do_training), "model do_training is not callable!"
            self.global_env.run_function(model.do_training)
            return

        for _ in range(self.n_epochs):
            self.epoch()
            self.epoch_n += 1
        
        self.global_env.record("training_finished", True)
        for hook in self.end_hooks:
            hook()

    @staticmethod
    def move_batch_to(batch, device, non_blocking=False):
        match batch:
            case batch if hasattr(batch, "to"):
                return batch.to(device, non_blocking=non_blocking)
            case batch if hasattr(batch, "_asdict"):
                t_ = type(batch)
                return t_(**{key: value.to(device, non_blocking=non_blocking) 
                             for key, value in batch._asdict().items()})
            case dict():
                return {key: value.to(device, non_blocking=non_blocking) 
                        for key, value in batch.items()}
            case _ if isinstance(batch, torch.Tensor):
                return batch.to(device, non_blocking=non_blocking)
            case (*seq,):
                return type(seq)(i.to(device, non_blocking=non_blocking) 
                                 for i in seq)
            case _:
                raise ValueError(f"Couldn't find how to move object {batch} to device {device}")
    
    @staticmethod
    def get_optimizer(name: str|Type[torch.optim.Optimizer], model_parameters, optimizer_arguments):
        if isinstance(name, str):
            optimizer_type = vars(torch.optim)[name]
        else: optimizer_type = name
        assert isinstance(optimizer_type, type)
        assert issubclass(optimizer_type, torch.optim.Optimizer)
        return optimizer_type(model_parameters, **optimizer_arguments)

    def get_optimizer_hook(self, name: str|Type[torch.optim.Optimizer], optimizer_arguments, clip_grad_norm=None, fake_batch_size=1) -> OptimizerHook:
        optimizer = self.get_optimizer(name, self.model.parameters(), optimizer_arguments)
        return OptimizerHook(optimizer, clip_gradient=clip_grad_norm, interval=fake_batch_size)

    @staticmethod
    def get_lr_scheduler(name: str, optimizer, scheduler_arguments):
        scheduler_type = vars(torch.optim.lr_scheduler)[name]
        return scheduler_type(optimizer, **scheduler_arguments)

    def get_lr_scheduler_hook(self, name: str, optimizer, scheduler_arguments):
        scheduler = self.get_lr_scheduler(name, optimizer, scheduler_arguments)
        return LRSchedulerHook(scheduler)

    def set_database(self, database_session: "DBSession", experiment: "DBExperiment", resume_from: "int|DBTraining_run|None",):
        from ..experiment_tracking import Training_run as DBTraining_run
        match resume_from:
            case DBTraining_run(database_object):
                self.id = database_object.id
                self.database_object = resume_from
            case int(id):
                self.id = id
                self.database_object = database_session.get(DBTraining_run, id)
            case None:
                dbtraining_run = self.get_database_object(experiment=experiment, database_session=database_session)
                database_session.add(dbtraining_run)
                database_session.commit()
                self.id = dbtraining_run.id
                self.database_object = dbtraining_run

    def get_database_object(self, experiment: "DBExperiment", database_session: "DBSession"):
        from ..experiment_tracking import Training_run as DBTraining_run
        model_db = self.model.get_database_object(database_session)
        if model_db is None:
            raise ValueError("Model doesn't have an id or a name, couldn't register to database")
        return DBTraining_run(
            model= model_db,
            training_parameters=self.training_parameters.model_dump(),
            experiement=experiment, 
        )

    def get_database_hook(self, checkpoint_interval: int = 100, commit_interval: int = 100, loss_name="loss", metrics=[] ):
        if self.database_session is None:
            raise ValueError("cannot get database hook if session is None")
        return DatabaseHook(database_session=self.database_session, checkpoint_interval=checkpoint_interval, commit_interval=commit_interval, loss_name=loss_name, metrics=metrics)



