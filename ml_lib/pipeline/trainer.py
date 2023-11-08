from collections.abc import Sequence
from collections import namedtuple
from typing import Type, Any, Union, Optional, Callable

from logging import info

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from ..models import load_model
from ..models.base_classes import Model
from ..datasets import load_dataset
from .training_hooks import TrainingHook
from ..environment import Environment, HierarchicEnvironment


class Trainer():

    model: Model
    optimizer: optim.Optimizer
    data: DataLoader
    
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

    def __init__(self, model: Model, 
                 data: Union[Dataset, DataLoader, Sequence],
                 n_epochs:int,
                 optimizer: Type[optim.Optimizer] | str = optim.Adam,
                 optimizer_arguments: dict[str, Any] = {}, 
                 loss = None,
                 lr_scheduler = None, 
                 device = "cuda:0", 
                 step_hooks: list[TrainingHook] = [], 
                 epoch_hooks: list[TrainingHook] = [],
                 environment_variables: dict = {}, 
                 fake_batch_size:int = 1
                 ):
        #TODO: take a train and validation set, or do the separation in-house

        device = torch.device(device)

        if lr_scheduler is not None:
            raise NotImplemented
        match data:
            case DataLoader():
                pass
            case Dataset():
                raise NotImplementedError
        if not isinstance(data, DataLoader):
            raise NotImplemented #TODO wrap it in a DataLoader
        if isinstance(optimizer, str):
            optimizer = vars(torch.optim)[optimizer]
        assert isinstance(optimizer, type)
        assert issubclass(optimizer, torch.optim.Optimizer)

        ################ Setup Environments
        self.global_env = Environment()
        self.global_env.record_dict(environment_variables)
        self.epoch_env = HierarchicEnvironment(parent=self.global_env)
        self.iter_env = HierarchicEnvironment(parent=self.epoch_env)

        ################ Actual training stuff
        self.optimizer = optimizer(params=model.parameters(), **optimizer_arguments)
        self.model = model.to(device)
        self.loss = loss; 
        self.data = data
        self.total_iter = len(data) * n_epochs
        self.n_epochs = n_epochs
        self.fake_batch_size = fake_batch_size
        self.device = device


        ################ set hooks
        for hook in step_hooks:
            hook.set_environment(self.iter_env)
        for hook in epoch_hooks:
            hook.env.set_environment(self.epoch_env)
        self.step_hooks = step_hooks
        self.epoch_hooks = epoch_hooks


        ################ Record stuff in environment
        if loss is not None:
            self.global_env.record("loss_fun", loss)
        self.global_env.record_dict(dict(
            model=model,
            total_iter = self.total_iter,
            n_epochs=n_epochs,
            device=device, 
            data=self.data
        ))
    @classmethod
    def from_config(cls, config: dict):
        raise NotImplementedError #TODO


    def step(self, batch):
        self.iter_env.reset()
        self.iter_env.record_dict(dict(
            iteration=self.iteration_n
        ))
        self.model.train()
        self.optimizer.zero_grad()

        batch = self.move_batch_to(batch, self.device)
        match batch:
            case dict():
                self.iter_env.record_dict(batch)
            case batch if hasattr(batch, "_asdict"):
                self.iter_env.record_dict(batch._asdict())
            case (x, gt):
                self.iter_env.record("x", x)
                self.iter_env.record("gt", gt)
            case _:
                self.iter_env.record("x", batch)
        self.iter_env.record("batch", batch)

        loss = self.iter_env.run_function(self.model.compute_loss, 
                                          record_result_as="loss")
        loss.backward()
        
        if (self.iteration_n - 1) % self.fake_batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

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

    @staticmethod
    def move_batch_to(batch, device):
        match batch:
            case batch if hasattr(batch, "_asdict"):
                t_ = type(batch)
                return t_(**{key: value.to(device) for key, value in batch._asdict().items()})
            case dict():
                return {key: value.to(device) for key, value in batch.items()}
            case _ if isinstance(batch, torch.Tensor):
                return batch.to(device)
            case (*seq,):
                return type(seq)(i.to(device) for i in seq)
            case _:
                return batch.to(device)

