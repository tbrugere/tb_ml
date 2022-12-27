from ..models import load_model
from ..models.base_classes import Unsupervised
from ..datasets import load_dataset

from collections.abc import Sequence
from typing import Type, Any, Union, Optional, Callable

from dataclasses import dataclass
from io import StringIO
from logging import info

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils import tensorboard
from tqdm import tqdm

@dataclass
class TrainingHook():

    interval: int = 1

    def __call__(self, **kwargs) -> dict:
        n_iteration = kwargs["iteration"]

        new_values = None
        if n_iteration % self.interval == 0:
            new_values = self.hook(**kwargs)

        if new_values is None:
            new_values = {}

        return {**kwargs, **new_values}

    def hook(self, **kwargs) -> Optional[dict]:
        pass


class Trainer():

    model: nn.Module
    optimizer: optim.Optimizer
    data: DataLoader
    
    """The eventual loss function"""
    loss: Optional[Callable]

    """The total number of iterations to run (ie n_epochs * len(data))"""
    total_iter: int

    """The number of the current iteration"""
    iteration_n: int = 0


    def __init__(self, model: nn.Module, 
                 data: Union[Dataset, DataLoader, Sequence],
                 n_epochs:int,
                 optimizer: Type[optim.Optimizer] | str = optim.Adam,
                 optimizer_arguments: dict[str, Any] = {}, 
                 loss = None,
                 lr_scheduler = None, 
                 device = "cuda:0", 
                 step_hooks: list[TrainingHook] = [], 
                 epoch_hooks: list[TrainingHook] = [],
                 ):
        #TODO: take a train and validation set, or do the separation in-house

        device = torch.device(device)

        if lr_scheduler is not None:
            raise NotImplemented
        if not isinstance(data, DataLoader):
            raise NotImplemented #TODO wrap it in a DataLoader and split it ?
        if isinstance(optimizer, str):
            optimizer = vars(torch.optim)[optimizer]
        assert isinstance(optimizer, type)
        assert issubclass(optimizer, torch.optim.Optimizer)

        self.optimizer = optimizer(params=model.parameters(), **optimizer_arguments)
        self.model = model.to(device)
        self.loss = loss
        self.data = data

        self.n_epochs = n_epochs

        self.step_hooks = step_hooks
        self.epoch_hooks = epoch_hooks

        self.total_iter = len(data) * n_epochs

        self.device = device

    @classmethod
    def from_config(cls, config):
        raise NotImplemented #TODO


    def step(self, batch, gt=None):
        self.model.train()
        self.optimizer.zero_grad()
        batch = batch.to(self.device)

        if gt is not None:
            gt = gt.to(self.device)

        match (self.loss, gt):
            case (None, None):
                prediction=None
                loss = self.model.forward(batch, loss=True) #unsupervised
            case (None, _):
                prediction=None
                loss = self.model.forward(batch, gt) #supervised, outputs loss
            case (loss_function, None):
                prediction = self.model.forward(batch, loss=False) #unsupervised
                loss = loss_function(prediction, batch)
            case _:
                prediction = self.model.forward(batch)# supervised, outputs prediction
                loss = self.loss(prediction, gt)

        loss.backward()

        self.optimizer.step()

        hook_parameters = {
                "loss": loss, 
                "prediction": prediction, 
                "iteration": self.iteration_n,
                "total_iter": self.total_iter,
                "trainer": self, #trainer passes itself. 
                                # this allows hooks to modify trainer parameters
                                # eg for learning rate scheduling
                "type": "step" #for hooks that act differently on step/epoch
            }

        for hook in self.step_hooks:
            hook_parameters = hook(**hook_parameters)

    def epoch(self):
        for point in self.data:
            if isinstance(self.model, Unsupervised):
                self.step(point)
                continue
            batch, gt = point
            self.step(batch, gt)

        hook_parameters = {
                "trainer": self, 
                "type": "epoch",
                "iteration": self.iteration_n,
                "total_iter": self.total_iter,
        }
        for hook in self.epoch_hooks:
            hook_parameters = hook(**hook_parameters)

    def train(self):
        model = self.model
        if hasattr(model, "do_pretraining"):
            info(f"Model {model} has do_pretraining method, launching")
            assert callable(model.do_pretraining), "model do_pretraining is not callable!"
            model.do_pretraining() #TODO pass arguments to that. EG DATA

        if hasattr(model, "do_training"):
            info(f"Model {model} has train method, using that")
            assert callable(model.do_training), "model do_training is not callable!"
            model.do_training() #TODO pass arguments to that. EG DATA, or hooks
            return

        for epoch_number in range(self.n_epochs):
            self.epoch()



class LoggerHook(TrainingHook):
    def hook(self, **kwargs):
        s = StringIO()
        for key, value in kwargs.items():
            if hasattr(value, "item") and value.numel() == 1:
                value = value.item()
            s.write(f"{key}= {value}, ")

        info(s.getvalue())


class CurveHook(TrainingHook):

    variable: str
    values: list

    def __init__(self, interval:int =1, variable="loss"):
        super().__init__(interval)
        self.variable = variable
        self.values = []

    def hook(self, **kwargs):
        val = kwargs[self.variable]
        if hasattr(val, "item"):
            val = val.item()
        self.values.append(val)

    def draw(self, ax = None): #todo: potentially output to file
        if ax is None:
            ax = plt.gca()
        values = self.values
        ax.set_title(self.variable)
        ax.set_ylabel(self.variable)
        ax.plot(np.arange(len(values)) * self.interval, values)




class TqdmHook(TrainingHook):
    progressbar: Optional[tqdm] = None

    def __init__(self, interval:int =1, tqdm=tqdm):
        super().__init__(interval)
        self.tqdm = tqdm

    def hook(self, **kwargs):
        if self.progressbar is None:
            totaliter =kwargs["total_iter"]
            self.progressbar = self.tqdm(total=totaliter)
        self.progressbar.update()

class TensorboardHook(TrainingHook):


