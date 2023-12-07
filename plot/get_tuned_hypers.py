import os
from pprint import pprint
import orbax
from flax.training import orbax_utils
import sparseeg.util.hyper as hyper
import numpy as np
import matplotlib.pyplot as plt
import click
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import yaml
import re


def inner_agg(x):
    return np.mean(x)


def outer_agg(x, axis):
    return np.mean(x, axis=axis)


def get_fname(data_file):
    if "dense" in data_file:
        alg = "dense"
    elif "set" in data_file:
        alg = "set"
    else:
        alg = "wp"

    n = re.findall(r'\d+', data_file)[-1]

    return f"{alg}_500epochs_{n}subject.yaml"


@click.argument("data_file", type=click.Path(exists=True))
@click.command
def get_hypers(data_file):
    chptr = orbax.checkpoint.PyTreeCheckpointer()

    print("Loading...")
    data = chptr.restore(data_file)
    data = hyper.satisfies(data, lambda x: x["dataset"]["batch_size"] == 8192)
    print("Loaded")

    print("Tuning")
    to_tune = "valid_accuracy"
    perfs = hyper.perfs(data, to_tune, inner_agg, outer_agg, combined=False)
    b = hyper.best(perfs, np.mean)
    config = data[str(b)]["config"]
    config["seed"] = list(range(3, 23))
    config["save_dir"] += "_next_20_seeds"

    config["model"]["activations"] = [config["model"]["activations"]]
    config["model"]["hidden_layers"] = [config["model"]["hidden_layers"]]

    if "weight_pruning" in data_file:
        args = config["model"]["optim"]["kwargs"]["scheduler"]["args"]
        config["model"]["optim"]["kwargs"]["scheduler"]["args"] = [args]

    pprint(config)

    fname = os.path.join(
        "src/sparseeg/config/eeg_low_final/weighted_loss/tuned_hypers/",
        get_fname(data_file),
    )
    print(get_fname(data_file))

    with open(fname, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == "__main__":
    get_hypers()
