import pandas as pd
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
import yaml

labels = []
predictions = []
percent = 1.0
n_subjects = 1
config_path = "./src/sparseeg/config/eeg_low_final/" + \
    "set_500epochs_{n_subjects}subject.yaml"

# Best hyper settings
if "dense" in config_path:
    if n_subjects = 3;
        ind = [3, 4, 5]
else:
    if n_subjects = 3;
        ind = [14, 15, 16]

with open(config_path, "r") as infile:
    config = yaml.safe_load(infile)
    full_config = config


# We'll need to average over seeds eventually
for i in range(len(ind)):
    index = ind[i]
    c, _ = hyper.sweeps(config, index)

    data_file = f"./results/{config['save_dir']}/{i}.pkl"
    chptr = orbax.checkpoint.PyTreeCheckpointer()

    print("Loading...")
    data = chptr.restore(data_file)
    print("Loaded")

    key = list(data.keys())[0]
    seed = list(data[key]["data"].keys())[0]
    model_state = data[key]["data"][seed]["model"]

    ds_config = config["dataset"]
    ds = default.dataset_fn(ds_config, c["seed"])

    # Get the testing dataset
    splitter = dataset.StratifiedTTV(
        c["train_percent"],
        c["valid_percent"],
        c["dataset"]["shuffle"],
        c["seed"],
    )
    _, ds, _ = splitter.split(ds)
    dl = loader.NumpyLoader(ds, len(ds), ds_config["shuffle"])

    # Load the model
    model = default.model_fn(data[key]["config"]["model"], 1, ds)

    # Compute label metrics
    for x_batch, y_batch in dl:
        batch = {"inputs": x_batch, "labels": y_batch}
        logits = model.apply(
            {'params': model_state["params"]}, batch['inputs'],
        )
        pred = jnp.argmax(logits, axis=-1)

        labels.extend(y_batch)
        predictions.extend(pred)

labels = np.array(labels).ravel()
predictions = np.array(predictions).ravel()

pp_matrix_from_data(labels, predictions, cmap="BuPu")
print(skmetrics.precision_recall_fscore_support(
    labels, predictions
))
