from typing import Optional, TYPE_CHECKING

from dataclasses import dataclass, field
from io import StringIO
from logging import info
import os
from pathlib import Path

if TYPE_CHECKING:
    import matplotlib.axes
    from tqdm import tqdm

from ..environment import HasEnvironmentMixin, Scope, scopevar_of_str, str_of_scopevar
from ..misc import find_file
from .annealing_scheduler import AnnealingScheduler, get_scheduler

@dataclass
class TrainingHook(HasEnvironmentMixin):

    interval: int = 1
    # env: Environment = field(default_factory=Environment)
    __post_init__ = HasEnvironmentMixin.__init__

    def __call__(self):
        n_iteration = self.env.iteration

        new_values = None
        if n_iteration % self.interval == 0:
            self.env.environment.run_function(self.hook)

        if new_values is None:
            new_values = {}

    def hook(self) -> Optional[dict]:
        raise NotImplementedError

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

    def __init__(self, interval:int =1, tqdm=None):
        super().__init__(interval)
        if tqdm is None:
            from tqdm.auto import tqdm
        self.tqdm = tqdm
        self.progressbar = None

    def hook(self):
        if self.progressbar is None:
            totaliter =self.env.total_iter
            self.progressbar = self.tqdm(total=totaliter)
        self.progressbar.update()

class TensorboardHook(TrainingHook):
    
    def __init__(self, tensorboard_dir:Optional[str] = None, interval: int=1, log_vars = ["loss"]):
        from torch.utils import tensorboard
        super().__init__(interval)
        if tensorboard_dir is None:
            tensorboard_path = find_file([Path("tensorboard"), Path("../tensorboard"), Path(f"{os.environ['HOME']}/tensorboard")])
            if tensorboard_path is None: tensorboard_dir = "tensorboard"
        else : tensorboard_path = Path(tensorboard_dir)
        self.tensorboard = tensorboard
        self.writer = tensorboard.SummaryWriter(tensorboard_path)
        self.log_vars = [scopevar_of_str(v) for v in log_vars]

    def hook(self):
        step = self.env.iteration
        loss_dict= dict()
        for scope, var in self.log_vars:
            loss_dict[var] = self.env.get(var, scope)
        self.writer.add_scalars('loss', loss_dict, step)


    
