from typing import Any

from copy import deepcopy
import functools as ft

from pydantic import (BaseModel, Field)
import yaml


from ml_lib.register import Loader
from ml_lib.models import register as model_register
from ml_lib.datasets import register as dataset_register, Dataset, transform_register
from .trainer import Training_parameters



class ModelConfig(BaseModel):
    type: str = Field()
    name: str = Field()
    params: dict[str, Any] = Field(default={})
    training_parameters: Training_parameters = Field(
            default=lambda: Training_parameters(n_epochs=1))

    def load_model(self, additional_params, register=model_register):
        model_class = register[self.type]
        return model_class()

class DatasetConfig(BaseModel):
    type: str = Field()
    params: dict[str, Any] = Field(default={})

    def load_dataset(self, additional_params, )




class ExperimentConfig(BaseModel):
    models: list[ModelConfig] = Field()
    datasets: dict[str, Any] = Field()
    test_dataset: dict[str, Any] = Field()

    @classmethod
    def load_yaml(cls, config_path):
        return cls.model_validate(yaml.safe_load(config_path.read_text()))

    class Config:
        ignored_types = (ft.cached_property,)
        protected_namespaces = ()


class Experiment():
    config: ExperimentConfig

    _datasets: dict[str, Dataset]
    _models_by_name: dict[int, Model]


    def __init__(self, config):
        self._datastes = {}

    def load_dataset(self, which="train", cache=True):
        if self._datasets is None:
            self._datasets = {}
        if which in self._datasets:
            return self._datasets[which]
        dataset_config = self.dataset if which == "train" else self.test_dataset
        dataset_config = {**dataset_config}
        transforms = dataset_config.pop("transforms", [])
        loader = Loader(dataset_register)
        # transform_loader = Loader(transform_register)

        transforms = [transform_register[t] for t in transforms]

        dataset = loader.load_config(dataset_config)

        for transform in transforms:
            dataset = transform(dataset)

        if cache:
            self._datasets[which] = dataset
        return dataset


    def load_models(self, return_training_params=False):
        models = []
        training_params = []
        dataset = self.load_dataset()
        # TODO: need a way to pull info from dataset to models
        for model_config in self.models:
            model_class = model_register[model_config.type]
            params = deepcopy(model_config.params)
            if vars(model_class).get("need_sizes", False):
                params["n_variables"] = dataset.n_variables()
                params["n_clauses"] = dataset.n_clauses()
            model = model_class(**params, name=model_config.name)
            models.append(model)
            training_params.append(model_config.training_params)

        if return_training_params:
            return models, training_params
        return models
