"""ml_lib.pipeline.control: Command line and eventually daemon to run experiments"""
from typing import TYPE_CHECKING, Self, TypeAlias, Literal
from os import PathLike
import functools as ft
from pathlib import Path
from logging import getLogger; log = getLogger("__main__")

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

    def __init__(self, experiment: PathLike, commands, *,
                 device: "str|torch.device|None"=None, 
                 database: PathLike, resume:str): 
        import torch
        self.experiment_config=Path(experiment)
        if device is None:
            raise NotImplementedError("need to implement device auto select")
        self.device = torch.device(device)
        self.commands = commands
        self.database = Path(database)
        self.resume = resume
        
    def run(self):
        with self.database_session() as db_session:
            exp = Experiment.from_yaml(self.experiment_config, 
                                       database_session=db_session)
            for command in self.commands:
                self.run_command(exp, command)

    def run_command(self, exp, command: CommandType):
        match command:
            case "train":
                exp.train_all(device=self.device, resume_from=self.resume)
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
                   resume=args.resume
                   )

    
    @staticmethod
    def argument_parser():
        from argparse import ArgumentParser
        from pathlib import Path
        import os

        parser = ArgumentParser()

        parser.add_argument("config", 
                            type=Path, )
        parser.add_argument("command", nargs="+", type=str)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--database", type=Path, 
                            default=os.environ.get("EXPERIMENT_DATABASE", "experiment_database.db"))
        parser.add_argument("--resume", type=str, default="highest_step")
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






