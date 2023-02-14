from typing import Optional, TYPE_CHECKING

from dataclasses import dataclass, field
from io import StringIO
from logging import info



if TYPE_CHECKING:
    import matplotlib.axes
    from tqdm import tqdm

from ..environment import Environment, Scope, scopevar_of_str, str_of_scopevar

@dataclass
class TrainingHook():

    interval: int = 1
    env: Environment = field(default_factory=Environment)

    def __call__(self):
        n_iteration = self.env.iteration

        new_values = None
        if n_iteration % self.interval == 0:
            self.env.run(self.hook)

        if new_values is None:
            new_values = {}

    def hook(self) -> Optional[dict]:
        raise NotImplementedError

class LoggerHook(TrainingHook):
    variables: list[tuple[Scope, str]]

    def __init__(self, variables: list[str] = [], interval=1):
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
        self.plt = plt

    def hook(self):
        val = self.env.get(self.variable, self.scope)
        if hasattr(val, "item"):
            val = val.item()
        self.values.append(val)

    def draw(self, ax: "matplotlib.axes.Axes|None" = None): #todo: potentially output to file
        import numpy as np
        if ax is None:
            ax = self.plt.gca()
            assert isinstance(ax, matplotlib.axes.Axes)
        values = self.values
        ax.set_title(self.variable)
        ax.set_ylabel(self.variable)
        ax.plot(np.arange(len(values)) * self.interval, values)




class TqdmHook(TrainingHook):
    progressbar: Optional["tqdm"] = None

    def __init__(self, interval:int =1, tqdm=None):
        super().__init__(interval)
        if tqdm is None:
            from tqdm import tqdm
        self.tqdm = tqdm
        self.progressbar = None

    def hook(self):
        if self.progressbar is None:
            totaliter =self.env.total_iter
            self.progressbar = self.tqdm(total=totaliter)
        self.progressbar.update()

class TensorboardHook(TrainingHook):
    
    def __init__(self):
        from torch.utils import tensorboard
        self.tensorboard = tensorboard

        raise NotImplementedError

