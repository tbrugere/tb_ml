# ml_lib

Tools for fast prototyping of machine learning models in pytorch.

This library is a collection of tools I have built over time while prototyping machine learning models in pytorch. 
This is not focused on production, and is intedend for my use-case, which is developing new models, with a focus on experimentation. 
In particular it has no support for multi-gpu or distributed training, or sharing model weights.
For a more complete tool, I recommend looking at [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/)

This is a work in progress and relatively unstable, 

## Features

- [x] Systematic definition of model hyperparameters from type definitions [see the Model class](ml_lib/models)
- [x] Basically everything from the command line interface to loading every training / model / dataset / etc. parameters from configs to the training code. Only model and dataset definitions are left to the user.
- [x] `Trainer` class for training models from configuration files, with automatic checkpointing, logging to database, and more…
- [x] Saving and loading models / parameters / experiments to a sqlite database
- [x] A lot of miscelanneous features such as printing model sizes as trees, keeping track of a model's device, freezing models, checking all given input to a model are on the right device, automatically creating tensors on the right device, etc.
- [ ] Job control for sending jobs to several machines / vms / cloud workers

## Installation

```commandline
$ pip install git+https://github.com/tbrugere/ml_lib
```


