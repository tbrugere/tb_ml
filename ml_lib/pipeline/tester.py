from typing import TYPE_CHECKING, Union, Sequence
from pydantic import BaseModel, Field
from logging import getLogger; log = getLogger(__name__)

if TYPE_CHECKING:
    import torch
    from sqlalchemy.orm import Session as DBSession
    from .experiment_tracking import Training_run as DBTraining_run, Experiment as DBExperiment, Training_step as DBTraining_step
    from ml_lib.datasets import Dataset

from torch.utils.data import DataLoader

from ml_lib.environment import Environment, HierarchicEnvironment
from ml_lib.models.base_classes import Model
from ml_lib.pipeline.testing.tests import Test


class Testing_parameters(BaseModel):
    max_n: int
    tests: list[str|dict]

    batch_size: int = 1


class Tester():
    model: Model
    data: DataLoader

    parameters: Testing_parameters

    id: int|None = None
    """id in the database"""
    database_object: "DBTraining_run|None" = None

    database_session: "DBSession|None"

    global_env: Environment
    iter_env: Environment

    def __init__(self, model: Model, 
                 data: Union[Dataset, DataLoader, Sequence],
                 testing_parameters: Testing_parameters, 
                 device: str|torch.device = "cuda:0", 
                 environment_variables: dict = {}, 
                 database: "DBSession|None" = None, 
                 db_experiment: "DBExperiment|int|None" = None, 
                 ):
        import torch

        self.parameters = testing_parameters
        self.device=torch.device(device)

        match data:
            case DataLoader():
                pass
            case Dataset():
                data = self.get_dataloader(data)

    def test_element(self,):

    def test_batch(self,):

    def test(self):



    def get_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset, 
            batch_size=self.parameters.batch_size, 
            collate_fn=dataset.collate, 
            shuffle=False, # Thatk's why
            pin_memory= self.parameters.performance_tricks 
                            and self.device.type == "cuda", 
            num_workers=5 if self.parameters.performance_tricks else 0,
            )
