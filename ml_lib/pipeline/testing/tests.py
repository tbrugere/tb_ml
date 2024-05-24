from typing import ClassVar
import numpy as np

from ml_lib.environment import HasEnvironmentMixin
from ml_lib.datasets import Datapoint
from ml_lib.register import Register

register = Register

class Test(HasEnvironmentMixin):
    title: ClassVar[str]

class SupervisedTest(Test):

    title: ClassVar[str]

    def compute(self, input_datapoint: Datapoint, results) -> np.ndarray:
        del input_datapoint, results
        raise NotImplementedError("This is a base class")

