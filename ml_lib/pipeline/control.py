"""ml_lib.pipeline.control: Command line and eventually daemon to run experiments"""
import logging
from typing import TYPE_CHECKING, ContextManager, Self, TypeAlias, Literal
import contextlib
import functools as ft
import os
from os import PathLike
from pathlib import Path
from logging import getLogger
import typing; log = getLogger("__main__")


from ml_lib.misc.context_managers import set_num_threads, set_log_level, torch_profile_auto_save, torch_set_sync_debug_mode

if TYPE_CHECKING:
    import torch
    import argparse

from ml_lib.pipeline.experiment import Experiment, ExperimentConfig

def set_sqlite_wal2(engine):
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def enable_wal(dbapi_connection, connection_record):
        log.info("Opening connection in WAL2 mode")
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL") #TODO WAL2??? when it's a thing
        cursor.close()

    @event.listens_for(engine, "close")
    def _(dbapi_connection, connection_record):
        log.info("Connection was closed. running pragma optimize")
        dbapi_connection.execute("PRAGMA optimize;") 
        dbapi_connection.execute("PRAGMA incremental_vacuum;") 


def get_database_engine(database_location):
    from sqlalchemy import create_engine, URL
    from sqlalchemy.orm import Session
    from ml_lib.pipeline.experiment_tracking import create_tables
    db_engine = create_engine(URL.create("sqlite", database=str(database_location)))
    set_sqlite_wal2(db_engine)
    create_tables(db_engine)
    return db_engine

CommandType: TypeAlias = Literal["train", "cleanup", "status"]
command_choices = typing.get_args(CommandType)

class CommandLine():

    experiment_config: Path
    device: "torch.device"
    commands: list[str]
    database: Path
    resume: str
    only_model: str|int|None = None

    # debugging and utilities
    log_level: int
    profile: bool = False
    debug_sync: bool = False

    def __init__(self, experiment: PathLike, commands, *,
                 device: "str|torch.device|None"=None, 
                 database: PathLike, resume:str, 
                 only_model: str|int|None = None, 
                 log_level: int = logging.WARNING, 
                 profile: bool= False, 
                 debug_sync: bool = False): 
        import torch
        self.experiment_config=Path(experiment)
        if device is None:
            raise NotImplementedError("need to implement device auto select")
        self.device = torch.device(device)
        self.commands = commands
        self.database = Path(database)
        self.resume = resume
        try:
            only_model_ = int(only_model)#type: ignore
            only_model = only_model_
        except: pass
        self.only_model = only_model
        self.log_level = log_level
        self.profile = profile
        self.debug_sync = debug_sync
        
    def run(self):
        with self.database_session() as db_session:
            exp = Experiment.from_yaml(self.experiment_config, 
                                       database_session=db_session)
            with self.running_context(): # sets everything up nicely
                for command in self.commands:
                    self.run_command(exp, command)#type: ignore

    def run_command(self, exp: Experiment, command: CommandType):
        match command:
            case "train":
                if self.only_model is None:
                    exp.train_all(device=self.device, resume_from=self.resume)
                else: 
                    exp.train(self.only_model, device=self.device, resume_from=self.resume)
            case "cleanup":
                self.cleanup_database()
            case "status":
                exp.print_status()
            case "_":
                raise NotImplementedError(f"Unsupported command {command}")
        
    def database_session(self):
        from sqlalchemy.orm import Session
        db_engine = get_database_engine(self.database)
        # we disable autoflush because it is a pain when doing concurrency (the database gets locked all the time)
        # now the database only gets locked when flushing, so if we only flush using commit(), we 
        # immediately unlock it afterwards
        return Session(db_engine, autoflush=False) 

    @classmethod
    def from_commandline(cls, args=None) -> Self:
        if args is None:
            argument_parser = cls.argument_parser()
            args = argument_parser.parse_args()
        return cls(args.config, 
                   commands=args.command, 
                   device=args.device, 
                   database=args.database, 
                   resume=args.resume, 
                   only_model = args.only_model, 
                   log_level= args.loglevel, 
                   profile= args.profile, 
                   debug_sync= args.debug_sync
                   )

    
    @classmethod
    def argument_parser(cls, parser: "argparse.ArgumentParser|None"=None):
        import argparse
        from pathlib import Path
        import os

        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("config", 
                            type=Path, )
        parser.add_argument("command", nargs="+", type=str, 
                            choices=command_choices)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--database", type=Path, 
                            default=os.environ.get("EXPERIMENT_DATABASE", "experiment_database.db"))
        parser.add_argument("--resume", type=str, default="highest_step", 
                            choices=Experiment.resume_from_options)
        parser.add_argument("--only-model", type=str, default=None)
        parser.add_argument(
            '-d', '--debug',
            help="Print lots of debugging statements",
            action="store_const", dest="loglevel", const=logging.DEBUG,
            default=logging.WARNING,
        )
        parser.add_argument(
            '-v', '--verbose',
            help="Be verbose",
            action="store_const", dest="loglevel", const=logging.INFO,
        )

        parser.add_argument(
            "--profile",
            help="use profiler", 
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--debug-sync",
            help="use synchronization debug mode", 
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        return parser


    def cleanup_database(self, dryrun=False):
        from ml_lib.pipeline.experiment_tracking import Checkpoint, Training_step
        from sqlalchemy import delete, not_ , func, text, select
        from datetime import timedelta
        with self.database_session() as session:
            should_delete = select(Checkpoint.id)\
                .join(Training_step)\
                .where(func.julianday(func.now()) - func.julianday(Training_step.step_time) > 2)\
                .where(not_(Checkpoint.is_last))
            number_to_delete, = session.execute(select(func.count(should_delete.cte().c.id))).one()

            log.info(f"deleting {number_to_delete} elements...")
            query = delete(Checkpoint)\
                    .where(Checkpoint.id.in_(should_delete))
            log.info("executing delete query...")
            session.execute(query, execution_options={"synchronize_session": False})
            session.commit()
            log.info("vacuuming database...")
            session.execute(text("VACUUM"))


    @contextlib.contextmanager
    def running_context(self):
        """Activates some context managers, and sets some defaults.
        This is what takes care of 
        - logging
        - profiling
        - setting the number of torch threads
        - etc."""
        managers: list[ContextManager] = []

        # maximum number of CPU threads
        n_threads = os.environ.get("NUM_THREADS")
        if n_threads is not None:
            managers.append(set_num_threads(int(n_threads)))

        # nice handling of logging with tqdm
        try: from tqdm.contrib.logging import logging_redirect_tqdm
        except ImportError: pass
        else: managers.append(logging_redirect_tqdm())

        # set logging level
        managers.append(set_log_level(self.log_level))

        #eventually setup profiling
        if self.profile:
            from torch.profiler import profile, record_function, ProfilerActivity
            profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
            managers.append(torch_profile_auto_save(profiler, "trace.json"))

        # eventually setup torch sync debug mode
        if self.debug_sync:
            managers.append(torch_set_sync_debug_mode())

        with contextlib.ExitStack() as stack:
            for m in managers:
                stack.enter_context(m)
            yield


class CommandLineExternal(CommandLine):
    @classmethod
    def argument_parser(cls, parser: "argparse.ArgumentParser|None"=None):
        from argparse import ArgumentParser
        if parser is None:
            parser = ArgumentParser()
        parser.add_argument("package", type=str)
        parser = super().argument_parser(parser)
        return parser

    @classmethod
    def from_commandline(cls, args=None) -> Self:
        from pathlib import Path
        from importlib import import_module
        from importlib.util import spec_from_file_location, module_from_spec
        if args is None:
            args = cls.argument_parser().parse_args()
        package = args.package
        if Path(package).exists():
            spec = spec_from_file_location(package, package)
            if spec is None:
                raise ValueError(f"Could not load package {package}")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
        else: # try importing from path
            module = import_module(package)
        # TODO: get registers from the package
        # eg module.dataset_register , module.transform_register, module.model_register ...
        # and pass them down to the experiment
        return super().from_commandline(args)

def run_pipeline():
    CommandLineExternal.from_commandline().run()
