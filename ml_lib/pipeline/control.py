"""ml_lib.pipeline.control: Command line and eventually daemon to run experiments"""
from typing import TYPE_CHECKING, Self, TypeAlias, Literal
from os import PathLike
import functools as ft
from pathlib import Path

if TYPE_CHECKING:
    import torch

from ml_lib.pipeline.experiment import Experiment, ExperimentConfig

CommandType: TypeAlias = Literal["train"]

class CommandLine():

    experiment_config: Path
    device: "torch.device"
    commands: list[str]
    database: Path

    def __init__(self, experiment: PathLike, commands, *,
                 device: "str|torch.device|None"=None, 
                 database: PathLike): 
        import torch
        self.experiment_config=Path(experiment)
        if device is None:
            raise NotImplementedError("need to implement device auto select")
        self.device = torch.device(device)
        self.commands = commands
        self.database = Path(database)
        
    def run(self):
        with self.database_session() as db_session:
            exp = Experiment.from_yaml(self.experiment_config, 
                                       database_session=db_session)
            for command in self.commands:
                self.run_command(exp, command)

    @staticmethod
    def run_command(exp, command: CommandType):
        match command:
            case "train":
                exp.train_all()
            case "_":
                raise NotImplementedError(f"Unsupported command {command}")
        
    def database_session(self):
        from sqlalchemy import create_engine, URL
        from sqlalchemy.orm import Session
        from ml_lib.pipeline.experiment_tracking import create_tables
        db_engine = create_engine(URL.create("sqlite", database=str(self.database)))
        create_tables(db_engine)
        return Session(db_engine)

    @classmethod
    def from_commandline(cls) -> Self:
        argument_parser = cls.argument_parser()
        args = argument_parser.parse_args()
        return cls(args.config, 
                   commands=args.command, 
                   device=args.device, 
                   database=args.database
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
        return parser



