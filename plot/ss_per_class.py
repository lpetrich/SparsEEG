# Plots highest posterior density credible interval with posterior mean
# (equivalently, the probability of getting a TP)
# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.64.142&rep=rep1&type=pdf

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import pymc3
import pandas as pd
import json
from sparseeg import dataset
import scipy
from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
import jax.numpy as jnp
from sparseeg.experiment import default as default
from flax.training import train_state
from sparseeg.training import state as training
import jax
import flax.linen as nn
import sparseeg.data.dataset as dataset
import sparseeg.data.loader as loader
from sparseeg.approximator.mlp import MLP
import os
from pprint import pprint
import orbax
from flax.training import orbax_utils
import sparseeg.util.hyper as hyper
import numpy as np
import matplotlib.pyplot as plt
import click
import pickle


def get_train_percents(data):
    p = set()
    for k in data.keys():
        p.add(data[k]["config"]["train_percent"])
    return list(p)


def inner_agg(x):
    return np.mean(x)


def outer_agg(x, axis):
    return np.mean(x, axis=axis)


def get_seeds(data):
    seeds = set()
    for key in data.keys():
        for s in data[key]["data"]:
            seed = "".join(filter(str.isdigit, s))
            seeds.add(int(seed))
    return list(seeds)


@click.argument("data_file")
@click.argument("seed")
@click.argument("p")
@click.command()
def get(p, seed, data_file):
    chptr = orbax.checkpoint.PyTreeCheckpointer()
    p = float(p)
    seed = int(seed)

    print("Loading...", data_file)
    data = chptr.restore(data_file)
    print("Loaded")

    new_data = hyper.satisfies(
        data,
        lambda x: (
            x["train_percent"] == p and
            x["dataset"]["batch_size"] == 8192
        ),
    )

    if len(new_data.keys()) > 1:
        raise IndexError()
    h = int(list(new_data.keys())[0])

    seed_key = f"seed_{seed}"
    print(str(h))
    ds_data = new_data[str(h)]["data"][seed_key]["dataset"]["test"]
    ds_config = {
        "type": "WAYEEGGALDataset-low",
        "n_subjects": 1,
        "empty": True,
    }
    ds = dataset.load(ds_config["type"], ds_config, seed)
    ds.data = ds_data
    dl = loader.NumpyLoader(ds, len(ds), False)

    model = default.model_fn(
        new_data[str(h)]["config"]["model"], seed, ds,
    )
    model_state = new_data[str(h)]["data"][seed_key]["model"]

    # Compute label metrics
    predictions = []
    labels = []
    for x_batch, y_batch in dl:
        batch = {"inputs": x_batch, "labels": y_batch}
        logits = model.apply(
            {'params': model_state["params"]}, batch['inputs'],
        )

        pred = jnp.argmax(logits, axis=-1)
        labels.extend(y_batch)
        predictions.extend(pred)

    p_r_f = skmetrics.precision_recall_fscore_support(
        labels, predictions,
    )

    if "dense" in data_file.lower():
        name = "dense"
    elif "set" in data_file.lower():
        name = "set"
    else:
        name = "wp"

    with open(f"plot_data/{name}_{seed}_{str(p)}.pkl", "wb") as outfile:
        pickle.dump(p_r_f, outfile)


if __name__ == "__main__":
    get()
