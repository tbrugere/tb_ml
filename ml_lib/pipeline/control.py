"""ml_lib.pipeline.control: Command line and eventually daemon to run experiments"""

from ml_lib.pipeline.experiment import Experiment


class CommandLine():

    experiment: Experiment
    
    @staticmethod
    def argument_parser():
        from argparse import ArgumentParser
        from pathlib import Path

        import torch

        parser = ArgumentParser()

        parser.add_argument("config", 
                            type=Path, 
                            required=True)

        parser.add_argument("commands", 
                            )

        parser.add_argument("--device", type=torch.device, default=None)


        




