"""
KL decay rate schedulers for KL annealing.
Following [@fuCyclicalAnnealingSchedule2019]

"""
from math import exp, cos, pi

#TODO: change that into a register
def get_scheduler(name: str, beta_0 = 1., T_0 = None, T_1 = None):
    scheduler = globals()[f"{name.capitalize()}Scheduler"]
    if name == "constant":
        return scheduler(beta_0)
    assert T_0 is not None

    if T_1 is None:
        return scheduler(beta_0, T_0)

    return CyclicScheduler(scheduler, T_0, T_1, beta_0)


class AnnealingScheduler():
    """
    Base class for KL-annealing schedulers

    A scheduler has a step() function, which returns the next value of the
    KL-loss rate.

    this class is a constant scheduler, takes an argument beta_0, and returns it with step()
    """

    beta_0: float = 1.

    def __init__(self, beta_0: float=1):

        """
        Args:
            beta_0: 
        """
        self.beta_0 = beta_0

    def step(self):
        return self.beta_0

    @classmethod
    def draw(cls, n_iterations: int, *args, **kwargs):
        """Draws this scheduler on current matpltolib ax 
        (use plt.sca if needed)

        Args:
            n_iterations: number
            *args, **kwargs: passed to the constructor 
        """
        import matplotlib.pyplot as plt
        scheduler = cls(*args, **kwargs)
        values = [scheduler.step() for _ in range(n_iterations)]
        plt.plot(values)


ConstantScheduler = AnnealingScheduler

class MonotonicScheduler(AnnealingScheduler):

    beta_0: float = 1.
    T_0: int = 1

    _n_steps:int = 0

    @staticmethod
    def f(x):
        """
        should be 0 in 0
        must be 1 in 1
        must be increasing monotonically

        Args:
            x: [TODO:description]
        """
        return 1

    def __init__(self, beta_0: float=1, T_0: int = 1):
        """Base class for KL-annealing Monotonic schedulers

        scheduler that increases monotonically until it reaches beta_0 
        at step T_0

        Its value is f(step / T_0) * beta_0 if step<= T_0
        beta_0 after that

        Args:
            lambda_0: 
            T_0:
        """
        self.beta_0 = beta_0
        self.T_0 = T_0
        self.reset()

    def step(self):
        beta_0 = super().step()
        self._n_steps += 1
        if self._n_steps >= self.T_0:
            return beta_0
        return beta_0 * self.f(self._n_steps / self.T_0)

    def reset(self):
        self._n_steps = 0


class CyclicScheduler(AnnealingScheduler):

    monotonic_scheduler: MonotonicScheduler

    def __init__(self, monotonic_scheduler, T_0, T_1, beta_0: float=1.):
        """[TODO:summary]

        Cyclic scheduler that takes T_0 to reach beta_0, and resets every T_1 steps

        Args:
            monotonic_scheduler: the underlying monotonic scheduler
            T_0: [TODO:description]
            T_1: [TODO:description]
            beta_0: [TODO:description]
        """
        super().__init__(beta_0)
        self.monotonic_scheduler = monotonic_scheduler(beta_0, T_0)
        self.T_1 = T_1

    def step(self):
        if self.monotonic_scheduler._n_steps >= self.T_1:
            self.monotonic_scheduler.reset()
        return self.monotonic_scheduler.step()


class LinearScheduler(MonotonicScheduler):
    @staticmethod
    def f(x):
        return x

class SigmoidScheduler(MonotonicScheduler):
    @staticmethod
    def f(x):
        return 2 * x / (1 + x) #not a sigmoid
        #but the only way I know that looks like a sigmoid and goes from 0 to 1
        #on [0, 1]


class CosineScheduler(MonotonicScheduler):
    @staticmethod
    def f(x):
        return 1 - cos(x * pi / 2)
