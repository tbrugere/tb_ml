import functools as ft
from collections.abc import Sequence
import contextlib
import time
from typing import Iterator, Type, Any, Union, Optional, Callable, TYPE_CHECKING, assert_never
from ml_lib.datasets.base_classes import Transform
from pydantic import BaseModel, Field, ConfigDict


import torch
from torch.utils.data import DataLoader

from ml_lib.misc.torch_functions import move_batch_to, precision_of_string
from ml_lib.datasets.utils import MultiEpochDataLoader
from ..models.base_classes import Model
from ..datasets import Dataset, Transform
from .training_hooks import TrainingHook, OptimizerHook, LRSchedulerHook, DatabaseHook
from ..environment import Environment, HierarchicEnvironment
from ..register import Loader
from .registers import loss_register, training_hook_register

from logging import getLogger; log = getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DBSession
    from .experiment_tracking import Training_run as DBTraining_run, Experiment as DBExperiment, Training_step as DBTraining_step

def try_tqdm(total: int, desc:str):
    try: from tqdm.auto import tqdm
    except ImportError: 
        log.info(desc)
        return range(total)
    else: return tqdm(range(total), desc=desc)

class Training_parameters(BaseModel):
    n_epochs: int = 1
    optimizer: str = "Adam"
    optimizer_arguments: dict =  Field(default_factory=dict)
    lr_scheduler: str|None = None
    loss: dict|None = None # to be gotten from a loss register

    batch_size: int|None = 10

    """fake_batch_size, or accumulation steps, 
    is the number of batches to run before doing an optimization step.
    Can be used to simulate a larger batch size, without using more memory.
    """
    fake_batch_size: int = 1
    
    """Gradient clipping. Default is 1, can be set to None to disable it"""
    clip_grad_norm: float|None = 1.

    """can be float32, float64, bfloat16... or "mixed" to use  torch.autocast"""
    precision: str|None = None 

    """"""

    step_hooks: list[dict|str] = []
    epoch_hooks: list[dict|str] = []
    end_hooks: list[dict|str] = []

    environment_variables: dict[str, Any] = Field(default_factory=dict)
    """Sets variables readable in the trainer's Environment, 
    They may be used by hooks or the model, or passed to the loss function"""

    performance_tricks: bool = True
    """enables various optimizations. Set to false to help debugging"""

    num_workers: int = 10
    """number of workers for the DataLoader. 
    Only used if performance_tricks is True, ignored otherwise"""

    no_optimizer_hook: bool = False
    """if True, the trainer will not run the optimizer hook. 
    Useful if the model runs the optimizer manually (eg. in the do_step method)
    Note that an optimizer will still be created, and can be accessed in the environment 
    as "env.optim"
    """

    checkpoint_interval: int = 10000
    database_commit_interval: int = 100

    train_transforms: list[dict|str] = [] #can also be a Transform, but need to implement the __get_pydantic_core_schema__ method to handle that
    """data transforms applied only at training time"""

    """other model's checkpoint to start from
    in the form 
    - model_name
    - model_name:max_step_n
    - model_name:run_n:max_step_n
    """
    start_from_other: str|None = None

    model_config = ConfigDict(
        ignored_types = (ft.cached_property,), 
        protected_namespaces = (), 
        extra = "forbid", 
    )

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

    """Whether the next epoch should skip datapoints 
    (and the number that should be skipped) .
    Used when resuming from a checkpoint (to get to the same point)"""
    skip_n_datapoints: int|None = None

    """Whether the last iteration OOMed"""
    oomed: bool = False

    def __init__(self, model: Model, 
                 data: Union[Dataset, DataLoader, Sequence],
                 training_parameters: Training_parameters, 
                 device: str|torch.device = "cuda:0", 
                 step_hooks: list[TrainingHook] = [], 
                 epoch_hooks: list[TrainingHook] = [],
                 end_hooks: list[TrainingHook] = [],
                 environment_variables: dict = {}, 
                 database: "DBSession|None" = None, 
                 db_experiment: "DBExperiment|int|None" = None, 
                 resume_from: "int|DBTraining_run|None" = None, 
                 resume_step: "int|DBTraining_step|None" = None, 
                 ):
        #TODO: take a train and validation set, or do the separation in-house

        device = torch.device(device)
        self.device = device
        self.training_parameters = training_parameters
        self.epoch_n = 0
        self.iteration_n = 0
        self.oomed = False

        if training_parameters.lr_scheduler is not None:
            raise NotImplemented
        match data:
            case DataLoader():
                if training_parameters.train_transforms:
                    log.warning("Transforms provided, but data is already a DataLoader, ignoring transforms")
                pass
            case Dataset():
                data = self.get_dataloader(data)
        if not isinstance(data, DataLoader):
            raise NotImplemented #TODO wrap it in a DataLoader

        ################ Hooks that the trainer itself sets
        # (eg. the optimizer hook)
        additional_step_hooks = []
        additional_epoch_hooks = []
        additional_end_hooks = []

        ################ Setup Environments
        self.global_env = Environment()
        self.global_env.record_dict(environment_variables)
        self.global_env.record("env_type", "global")
        self.epoch_env = HierarchicEnvironment(parent=self.global_env)
        self.iter_env = HierarchicEnvironment(parent=self.epoch_env)

        ################ Actual training stuff
        self.model = model.to(device, dtype=self.get_dtype())
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
        if not training_parameters.no_optimizer_hook:
            additional_step_hooks.append(optimizer_hook)

        ############### Database stuff
        self.database_session=database
        if database is not None:
            assert db_experiment is not None
            # This is where the resume stuff should be 
            self.set_database(database_session=database, experiment=db_experiment, resume_from=resume_from)
            database_hook = self.get_database_hook()
            additional_step_hooks.append(database_hook)
            additional_end_hooks.append(database_hook)


        ################ Record stuff in environment
        if training_parameters.loss is not None:
            self.global_env.record("loss_fun", training_parameters.loss)
        self.global_env.record_dict(dict(
            model=model,
            n_epochs=self.n_epochs, 
            total_iter = self.total_iter,
            device=device, 
            data=self.data, 
            training_parameters=training_parameters, 
        ))
        self.global_env.record_dict(training_parameters.model_dump())
        if database is not None:
            self.global_env.record("training_run_id", self.id)
            self.global_env.record("training_run_db", self.database_object)

        ################ set hooks
        hook_loader = Loader(training_hook_register)
        step_hooks = [hook_loader(hook_config) for hook_config in training_parameters.step_hooks] + step_hooks + additional_step_hooks
        epoch_hooks = [hook_loader(hook_config) for hook_config in training_parameters.epoch_hooks] + epoch_hooks + additional_epoch_hooks
        end_hooks = [hook_loader(hook_config) for hook_config in training_parameters.end_hooks] + end_hooks + additional_end_hooks

        #TODO: add lr_scheduler
        for hook in step_hooks:
            hook.set_environment(self.iter_env)
            hook.setup()
        for hook in epoch_hooks:
            hook.set_environment(self.epoch_env)
            hook.setup()
        for hook in end_hooks:
            hook.set_environment(self.global_env)
            hook.setup()
        self.step_hooks = step_hooks
        self.epoch_hooks = epoch_hooks
        self.end_hooks = end_hooks

        ############## Eventually resume 
        if training_parameters.start_from_other is not None:
            self.load_checkpoint(training_parameters.start_from_other)

        if resume_from is not None:
            self.resume(resume_step)


    def try_step(self, batch):
        if self.oomed: self.recover_from_oom()
        self.oomed = False
        try: self.step(batch) 
        except RuntimeError as e:
            if "out of memory" in str(e): 
                self.oomed = True
            else: raise


    def step(self, batch):
        self.iter_env.reset()
        self.iter_env.record_dict(dict(
            env_type="iteration",
            iteration=self.iteration_n
        ))
        self.model.train()
        self.model.set_environment(self.iter_env)

        batch = move_batch_to(batch, self.device, 
                              dtype=self.get_dtype(),
                    non_blocking=self.training_parameters.performance_tricks)
        match batch:
            case dict():
                self.iter_env.record_dict(batch)
            case batch if hasattr(batch, "asdict"):
                self.iter_env.record_dict(batch.asdict())#type: ignore
            case batch if hasattr(batch, "_asdict"):
                self.iter_env.record_dict(batch._asdict())#type: ignore
            case (x, gt):
                self.iter_env.record("x", x)
                self.iter_env.record("gt", gt)
            case _:
                self.iter_env.record("x", batch)
        self.iter_env.record("batch", batch)

        with self.step_context():
            if self.model.do_step is not None:
                self.iter_env.run_function(self.model.do_step)
            else:
                loss = self.iter_env.run_function(self.model.compute_loss, 
                                                  record_result_as="loss")
                loss.backward()
        
        for hook in self.step_hooks:
            hook.set_environment(self.iter_env)
            hook()

    def epoch(self):
        self.epoch_env.reset()
        self.epoch_env.record_dict(dict(
            env_type="epoch",
            last_iteration=self.iteration_n, 
            epoch= self.epoch_n
        ))
        self.model.set_environment(self.epoch_env)

        data_iter = self.data_iter()

        if self.skip_n_datapoints is not None:
            self.skip_batches(self.skip_n_datapoints, data_iter)
            self.skip_n_datapoints = None

        for step_n, batch in data_iter:
            self.try_step(batch)
            self.iteration_n += 1

        for hook in self.epoch_hooks:
            hook.set_environment(self.epoch_env)
            hook()
    

    def train(self):
        model = self.model
        model.set_environment(self.global_env)
        log.info(f"Training model {model.model_name}")
        if model.do_warmup is not None:
            log.info(f"Model {model} has do_warmup method, launching")
            assert callable(model.do_warmup), "model do_warmup is not callable!"
            self.global_env.run_function(model.do_warmup)

        if model.do_pretraining is not None and self.iteration_n == 0:
            log.info(f"Model {model} has do_pretraining method, launching")
            assert callable(model.do_pretraining), "model do_pretraining is not callable!"
            self.global_env.run_function(model.do_pretraining)

        if model.do_training is not None:
            log.info(f"Model {model} has do_training method, using that")
            assert callable(model.do_training), "model do_training is not callable!"
            self.global_env.run_function(model.do_training)
            return

        while self.epoch_n < self.n_epochs:
            self.epoch()
            self.epoch_n += 1
        
        self.global_env.record("training_finished", True)
        for hook in self.end_hooks:
            hook.set_environment(self.global_env)
            hook()

    def load_checkpoint(self, checkpoint_str):
        from ml_lib.pipeline.experiment_tracking import Checkpoint
        assert self.database_session is not None, "cannot load checkpoint with no database session"
        checkpoint = Checkpoint.from_descriptor_string(checkpoint_str, self.database_session)
        if checkpoint is None: 
            log.warning(f'Could not load checkpoint "{checkpoint_str}": not found')
            return
        log.info(f"restarting from checkpoint of model {checkpoint.model.name} at step {checkpoint.step.step}")
        if not checkpoint.is_last:
            log.warning("Restarting from a non-final checkpoint!")
        self.model.load_checkpoint(checkpoint.checkpoint)

    def resume(self, step: "int|DBTraining_step|None"):
        session = self.database_session
        db_trainingrun = self.database_object
        assert session is not None
        assert db_trainingrun is not None
        ##### 1. get checkpoint (also sets the exact step number)
        if step is None:
            checkpoint = db_trainingrun.last_checkpoint()
            if checkpoint is None:
                log.warning("tried to resume, but did not find a checkpoint")
                return
        else: 
            step_num = step.step if isinstance(step, DBTraining_step) else step
            checkpoint = db_trainingrun.last_checkpoint(max_step_n=step_num)
            if checkpoint is None:
                log.warning("tried to resume, but did not find a checkpoint")
                return
            if checkpoint.step.step < step_num:
                log.warning(f"asked to resume from step {step_num}, but no checkpoint was found for that step")
                log.warning(f"using that from  {checkpoint.step.step} instead")
        log.info(f"resuming from step {checkpoint.step.step}") 
        self.model.load_checkpoint(checkpoint.checkpoint)
        step_num = checkpoint.step.step + 1# we saved *after* step step_num, so we restart at step_num + 1
        self.set_step_num(step_num)

    def set_step_num(self, step_num):
        """Sets the step number. Useful for resuming from a checkpoint"""

        epoch, epoch_step = divmod(step_num, len(self.data))

        self.iteration_n = step_num
        self.epoch_n = epoch

        self.skip_n_datapoints = epoch_step
        
        self.epoch_env.record_dict(dict(
            last_iteration=self.iteration_n, 
            epoch= self.epoch_n
        ))
        for hook in self.epoch_hooks:
            hook.set_environment(self.epoch_env)
            hook.set_state()

        self.iter_env.record("iteration", self.iteration_n)
        for hook in self.step_hooks:
            hook.set_environment(self.iter_env)
            hook.set_state()

    def recover_from_oom(self):
        import gc
        log.warning(f"OOM, skipping iteration {self.iteration_n - 1}")
        self.iter_env.reset()
        gc.collect()
        torch.cuda.empty_cache()

    def get_dtype(self):
        precision = self.training_parameters.precision
        if precision is not None and precision != "mixed":
            return precision_of_string(precision)
        else: return None

    @contextlib.contextmanager
    def step_context(self):
        """Enters the various context managers we might want to run a step in

        For now this is only :func:`torch.autocast` if the precision is set to "mixed"
        """
        with contextlib.ExitStack() as stack:
            if self.training_parameters.precision == "mixed":
                from torch import autocast
                stack.enter_context(autocast(device_type=self.device.type))
            yield

    @staticmethod
    def skip_batches(skip_n_steps: int, data_iter: Iterator):
        pbar = try_tqdm(skip_n_steps, desc=f"skipping {skip_n_steps} batches")
        for _ in zip(pbar, data_iter):
            pass

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
        self.global_env.record("optim", optimizer)
        return OptimizerHook(optimizer, clip_gradient=clip_grad_norm, interval=fake_batch_size)

    @staticmethod
    def get_lr_scheduler(name: str, optimizer, scheduler_arguments):
        scheduler_type = vars(torch.optim.lr_scheduler)[name]
        return scheduler_type(optimizer, **scheduler_arguments)

    def get_lr_scheduler_hook(self, name: str, optimizer, scheduler_arguments):
        scheduler = self.get_lr_scheduler(name, optimizer, scheduler_arguments)
        return LRSchedulerHook(scheduler)

    def set_database(self, database_session: "DBSession", experiment: "DBExperiment|int", resume_from: "int|DBTraining_run|None",):
        from .experiment_tracking import Training_run as DBTraining_run
        from .experiment_tracking import Experiment as DBExperiment
        if isinstance(experiment, int):
            maybe_experiment: DBExperiment|None =database_session.get(DBExperiment, experiment)
            assert maybe_experiment is not None, "provided experiment doesn't exist"
            experiment = maybe_experiment
        assert isinstance(experiment, DBExperiment)
        match resume_from:
            case DBTraining_run():
                self.id = resume_from.id
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
            case never:
                assert_never(never)
        self.global_env.record("database_object", self.database_object)

    def get_database_object(self, experiment: "DBExperiment", database_session: "DBSession"):
        from .experiment_tracking import Training_run as DBTraining_run
        model_db = self.model.get_database_object(database_session)
        if model_db is None:
            raise ValueError("Model doesn't have an id or a name, couldn't register to database")
        assert experiment in database_session
        return DBTraining_run(
            model= model_db,
            training_parameters=self.training_parameters.model_dump(),
            experiment_id=experiment.id, 
            n_steps = self.total_iter, 
        )

    def get_database_hook(self, loss_name="loss", metrics=[] ):
        if self.database_session is None:
            raise ValueError("cannot get database hook if session is None")
        return DatabaseHook(database_session=self.database_session, checkpoint_interval=self.training_parameters.checkpoint_interval, 
                            commit_interval=self.training_parameters.database_commit_interval, loss_name=loss_name, metrics=metrics)

    def get_dataloader(self, dataset: Dataset):
        if self.training_parameters.train_transforms:
            dataset = dataset.apply_transforms(self.training_parameters.train_transforms)
        if self.training_parameters.performance_tricks:
            data_loader_type = MultiEpochDataLoader
        else: data_loader_type = DataLoader
        return data_loader_type(
            dataset, 
            batch_size=self.training_parameters.batch_size, 
            collate_fn=dataset.collate, 
            shuffle=True,
            pin_memory= self.training_parameters.performance_tricks 
                            and self.device.type == "cuda", 
            num_workers=self.training_parameters.num_workers if self.training_parameters.performance_tricks else 0,
            persistent_workers=self.training_parameters.performance_tricks,
            )

    def data_iter(self):
        batch_feed_timer = time.perf_counter()
        for batch_idx, batch in enumerate(self.data):
            time_delta = time.perf_counter() - batch_feed_timer
            if time_delta > .05:
                log.warning(f"waited {time_delta:.2f}s for batch {batch_idx}")

            yield batch_idx, batch

            batch_feed_timer = time.perf_counter()


