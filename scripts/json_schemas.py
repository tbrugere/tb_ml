#!/bin/env python
"""Dump json schemas for ml_lib configs to stdout"""
from sys import stdout

def argument_parser():
    from argparse import ArgumentParser, BooleanOptionalAction
    from pathlib import Path
    import os

    parser = ArgumentParser()
    parser.add_argument("--which", type=str, default="experiment")
    parser.add_argument("--yaml", action=BooleanOptionalAction, default=True)
    return parser

def main(which_schema, use_yaml):
    match which_schema:
        case "experiment": 
            from ml_lib.pipeline.experiment import ExperimentConfig
            schema = ExperimentConfig.model_json_schema()
        case _:
            raise ValueError(f"Unrecognized schema {which_schema}")

    if use_yaml:
        import yaml
        yaml.dump(schema, stdout)
    else:
        import json
        json.dump(schema, stdout)



if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args.which, args.yaml)
