"""ml_lib.pipeline.control: Command line and eventually daemon to run experiments"""
import logging
from typing import TYPE_CHECKING, ContextManager, Self, TypeAlias, Literal
import contextlib
import functools as ft
import os
from os import PathLike
from pathlib import Path
from logging import getLogger; log = getLogger("__main__")

from ml_lib.misc.context_managers import set_num_threads, set_log_level, torch_profile_auto_save

if TYPE_CHECKING:
    import torch

from ml_lib.pipeline.experiment import Experiment, ExperimentConfig

def set_sqlite_wal2(engine):
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def enable_wal2(dbapi_connection, connection_record):
        log.info("Opening connection in WAL2 mode")
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL") #TODO WAL2???
        cursor.close()

def get_database_engine(database_location):
    from sqlalchemy import create_engine, URL
    from sqlalchemy.orm import Session
    from ml_lib.pipeline.experiment_tracking import create_tables
    db_engine = create_engine(URL.create("sqlite", database=str(database_location)))
    set_sqlite_wal2(db_engine)
    create_tables(db_engine)
    return db_engine

CommandType: TypeAlias = Literal["train", "cleanup", "status"]

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

    def __init__(self, experiment: PathLike, commands, *,
                 device: "str|torch.device|None"=None, 
                 database: PathLike, resume:str, 
                 only_model: str|int|None = None, 
                 log_level: int = logging.WARNING, 
                 profile: bool= False): 
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
        except ValueError: pass
        self.only_model = only_model
        self.log_level = log_level
        self.profile = profile
        
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
    def from_commandline(cls) -> Self:
        argument_parser = cls.argument_parser()
        args = argument_parser.parse_args()
        return cls(args.config, 
                   commands=args.command, 
                   device=args.device, 
                   database=args.database, 
                   resume=args.resume, 
                   only_model = args.only_model, 
                   log_level= args.loglevel, 
                   profile= args.profile
                   )

    
    @staticmethod
    def argument_parser():
        import argparse
        from pathlib import Path
        import os

        parser = argparse.ArgumentParser()

        parser.add_argument("config", 
                            type=Path, )
        parser.add_argument("command", nargs="+", type=str)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--database", type=Path, 
                            default=os.environ.get("EXPERIMENT_DATABASE", "experiment_database.db"))
        parser.add_argument("--resume", type=str, default="highest_step")
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
        return parser


    def cleanup_database(self):
        from ml_lib.pipeline.experiment_tracking import Checkpoint, Training_step
        from sqlalchemy import delete, not_ , func, text
        from datetime import timedelta
        with self.database_session() as session:
            query = delete(Checkpoint)\
                    .where(not_(Checkpoint.is_last))\
                    .where(func.now() - Checkpoint.step.step_time > timedelta(days=2) )
            session.execute(query)
            session.commit()
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

        with contextlib.ExitStack() as stack:
            for m in managers:
                stack.enter_context(m)
            yield





