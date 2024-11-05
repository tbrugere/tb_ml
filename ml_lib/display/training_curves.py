import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
from ml_lib.pipeline.experiment_tracking import Model as DBModel

from ml_lib.misc.data_structures import NotSpecified

def get_losses(training_run):
    return np.array([s.loss for s in training_run.steps if s.loss is not None])

def compute_running_mean(losses, rm_size):
    from scipy.ndimage import uniform_filter1d
    losses = losses.astype(np.float32)
    return uniform_filter1d(losses, size=rm_size,  mode="nearest", 
                       origin=rm_size // 2 -1) 

def display_loss_array(losses, running_means=[], steps=None, color="blue",
                      min=NotSpecified(), 
                      max=None, 
                      name=None, 
                      log=True,
                      grid=True, 
                      **drawing_args,
                      ):
    
    from matplotlib.colors import to_rgb
    losses = losses.astype(np.float32)

    r, g, b = to_rgb(color)
    a_step = 1.
    a_values = np.exp(- np.arange(len(running_means) + 1) * a_step)[::-1]

    if min is NotSpecified():
        if log: min = losses[losses == losses].min()
        else: min = 0
    if min is None: min = losses[losses == losses].min()
    if max is None: max = losses[losses==losses].max()
    plt.ylim((min, max))
    if log:
        plt.yscale("log")
    if grid:
        plt.grid(axis="y")

    plt.plot(losses, color=(r, g, b, a_values[0]), label=name, **drawing_args)
    for m, a in zip(running_means, a_values[1:]):
        if name is not None:
            curve_name = f"{name} - (Moving average {m})"
        else: curve_name=None
        plt.plot(compute_running_mean(losses, m), color=(r, g, b, a), label=curve_name)


def display_model_loss(model_name, db_engine, training_run_num=-1, **drawing_args):
    with Session(db_engine, autoflush=False) as session:
        db_model = DBModel.get_by_name(model_name, session)
        assert db_model is not None
        training_runs = db_model.training_runs
        tr = training_runs[training_run_num]
        losses = get_losses(tr)

    display_loss_array(losses, **drawing_args, name=model_name)

def display_losses(model_names, db_engine, color_cycle = None, **drawing_args):
    if color_cycle is None:
        color_cycle = plt.rcParams["axes.prop_cycle"]
    for m, c in zip(model_names, color_cycle):
        display_model_loss(m, db_engine, **c, **drawing_args)
    plt.ylabel("loss")
    plt.xlabel("steps")
    plt.legend()
    
