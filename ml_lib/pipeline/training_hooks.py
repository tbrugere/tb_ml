from typing import Optional, TYPE_CHECKING

from dataclasses import dataclass, field
from io import StringIO
from logging import info
import os
from pathlib import Path
from logging import getLogger; logger = getLogger(__name__)

if TYPE_CHECKING:
    import matplotlib.axes
    from tqdm import tqdm
    from sqlalchemy.orm import Session

from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_

from ..environment import HasEnvironmentMixin, Scope, scopevar_of_str, str_of_scopevar
from ..misc import find_file
from .annealing_scheduler import AnnealingScheduler, get_scheduler

@dataclass
class TrainingHook(HasEnvironmentMixin):

    interval: int|None = 1
    # env: Environment = field(default_factory=Environment)
    __post_init__ = HasEnvironmentMixin.__init__

    def __call__(self):
        if self.interval is None: 
            self.env.environment.run_function(self.hook) #this is for end hooks (only run once)
        n_iteration = self.env.iteration

        new_values = None
        if n_iteration % self.interval == 0:
            self.env.environment.run_function(self.hook)

        if new_values is None:
            new_values = {}

    def hook(self) -> Optional[dict]:
        raise NotImplementedError

    def set_state(self):
        """Set the state of the hook to the current state of the environment.
        Useful when the training is resumed from a checkpoint.
        Read the state of the environment and set the state of the hook accordingly.
        """
        pass

class EndHook(TrainingHook):
    def __init__(self):
        super().__init__(interval=None)
    def __call__(self):
        self.env.environment.run_function(self.hook) #no checking stuff because this is run once

class LoggerHook(TrainingHook):
    variables: list[tuple[Scope, str]]

    def __init__(self, variables: list[str] = ["loss"], interval=1):
        super().__init__(interval)
        self.variables = [scopevar_of_str(v) for v in variables]

    def hook(self):
        s = StringIO()
        for scope, key in self.variables:
            value = self.env.get(key=key, scope=scope)
            if hasattr(value, "item") and value.numel() == 1:
                value = value.item()
            s.write(f"{str_of_scopevar(scope, key)}= {value}, ")

        info(s.getvalue())

class CurveHook(TrainingHook):
    scope: Scope
    variable: str
    values: list

    def __init__(self, interval:int =1, variable="loss"):
        super().__init__(interval)
        self.scope, self.variable = scopevar_of_str(variable)
        self.values = []
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        self.plt = plt
        self.mpl = mpl

    def hook(self):
        val = self.env.get(self.variable, self.scope)
        if hasattr(val, "item"):
            val = val.item()
        self.values.append(val)

    def draw(self, ax: "matplotlib.axes.Axes|None" = None): #todo: potentially output to file
        import numpy as np
        if ax is None:
            ax = self.plt.gca()
            assert isinstance(ax, self.mpl.axes.Axes)
        values = self.values
        ax.set_title(self.variable)
        ax.set_ylabel(self.variable)
        ax.plot(np.arange(len(values)) * self.interval, values)

class KLAnnealingHook(TrainingHook):
    scope: Scope
    variable: str
    scheduler: AnnealingScheduler
    
    def __init__(self, variable="kl_coef",
                 scheduler: str|AnnealingScheduler = "constant",
                 beta_0 = None, T_0=None, T_1=None):
        super().__init__(interval=1)
        self.scope, self.variable = scopevar_of_str(variable)
        match scheduler:
            case AnnealingScheduler():
                assert beta_0 is None and T_0 is None and T_1 is None
                self.scheduler = scheduler
            case str():
                if beta_0 is None: beta_0 = 1.
                self.scheduler = get_scheduler(scheduler, beta_0, T_0, T_1)

    def hook(self):
        val = self.scheduler.step()
        self.env.record(self.variable, val, self.scope)

    def draw(self, ax: "matplotlib.axes.Axes|None" = None):
        self.scheduler.draw(ax=ax)

class TqdmHook(TrainingHook):
    progressbar: Optional["tqdm"] = None
    last_known_epoch: int = 0

    def __init__(self, interval:int =1, tqdm=None):
        super().__init__(interval)
        if tqdm is None:
            from tqdm.auto import tqdm
        self.tqdm = tqdm
        self.progressbar = None

    def hook(self):
        if self.progressbar is None:
            self.reset_progressbar()
        assert self.progressbar is not None
        if self.env.epoch != self.last_known_epoch:
            self.last_known_epoch = self.env.epoch
            self.progressbar.set_description(f"Epoch {self.env.epoch}")
        self.progressbar.update()

    def reset_progressbar(self, initial: int = 0):
        totaliter =self.env.total_iter
        epoch = self.env.epoch
        self.last_known_epoch = epoch
        self.progressbar = self.tqdm(total=totaliter, initial=initial, desc=f"Epoch {epoch}")

    def set_state(self):
        step = self.env.iteration
        self.reset_progressbar(initial=step)


class TensorboardHook(TrainingHook):
    
    def __init__(self, interval: int=1, *, tensorboard_dir:Optional[str] = None, run_name:str,  log_vars = ["loss"]):
        from torch.utils import tensorboard
        super().__init__(interval)
        if tensorboard_dir is None:
            tensorboard_path = find_file([Path("tensorboard"), Path("../tensorboard"), Path(f"{os.environ['HOME']}/tensorboard")])
            if tensorboard_path is None: tensorboard_path = Path("tensorboard")
        else : tensorboard_path = Path(tensorboard_dir)
        tensorboard_path = tensorboard_path / run_name
        self.tensorboard = tensorboard
        self.writer = tensorboard.SummaryWriter(str(tensorboard_path))
        self.log_vars = [scopevar_of_str(v) for v in log_vars]

    def hook(self):
        step = self.env.iteration
        loss_dict= dict()
        for scope, var in self.log_vars:
            loss_dict[var] = self.env.get(var, scope)
        self.writer.add_scalars('loss', loss_dict, step)


class OptimizerHook(TrainingHook):
    def __init__(self, optimizer: Optimizer, clip_gradient: Optional[float] =None, interval: int=1):
        super().__init__(interval)
        self.optimizer = optimizer
        self.clip_gradient = clip_gradient

    def hook(self):
        model = self.env.get("model")
        if self.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), self.clip_gradient)

        self.optimizer.step()
        self.optimizer.zero_grad()

class LRSchedulerHook(TrainingHook):
    def __init__(self, scheduler, interval: int=1):
        super().__init__(interval)
        self.scheduler = scheduler

    def hook(self):
        self.scheduler.step()



class DatabaseHook(TrainingHook):
    database_session: "Session"
    model_id: int
    training_run_id: int

    def __init__(self, interval: int=1, *, database_session: "Session", checkpoint_interval: int = 100, commit_interval: int = 1000):
        super().__init__(interval)
        self.database_session = database_session
        self.table = table

    def hook(self):
        """NOT IMPLEMENTED YET"""
        raise NotImplementedError
        from ..experiment_tracking import Model as DBModel, Training_run as DBTraining_run, Training_step as DBTraining_step
        self.database.insert(self.table, self.env)


class SlackHook(EndHook):
    token: str
    channel: str

    def __init__(self, *, token:str|None=None, channel:str|None=None, ):
        super().__init__()
        if token is None:
            token = os.environ["SLACK_TOKEN"]
        if channel is None:
            channel = os.environ["SLACK_CHANNEL"]
        if channel.startswith("#"):
            import requests
            response = requests.get("https://slack.com/api/conversations.list", params={
                "token": token
            })
            assert response.ok
            response = response.json()
            assert response["ok"]
            channels = response["channels"]
            channel_name = channel[1:]
            for channel_i in channels:
                if channel_i["name"] == channel_name:
                    channel = channel_i["id"]
                    break
            else:
                logger.error(f"SlackHook: Channel {channel_name} not found, will not send slack messages")


        assert channel is not None
        self.token = token
        self.channel = channel

    def hook(self):
        import requests
        model = self.env.get("model")
        model_name = model.model_name
        if model_name is not None:
            train_info = model_name
        else:
            import sys
            train_info = " ".join(sys.argv)

        requests.post("https://slack.com/api/chat.postMessage", data={
            "token": self.token,
            "channel": self.channel,
            "text": f"Training finished for {train_info}"
        })

        


