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

labels = []
predictions = []
percent = 1.0
batch_size = 30000
alg_type = "dense"
n_subjects = 3

# We'll need to average over seeds eventually
for i in range(3):
    if alg_type == "dense":
        if n_subjects == 3:
            inds = [3, 4, 5]
    else:
        if n_subjects == 3:
            inds = [14, 15, 16]

    if n_subjects == 3:
        subjects = [
            [1, 2, 3, 4, 5, 6, 10, 11, 12],
            [1, 2, 3, 4, 7, 8, 9, 11, 12],
            [1, 4, 5, 6, 7, 8, 10, 11, 12],
        ]

    data_file = f"./results/{alg_type}_500epochs_3subject/{inds[i]}.pkl"
    chptr = orbax.checkpoint.PyTreeCheckpointer()

    print("Loading...")
    data = chptr.restore(data_file)
    print("Loaded")

    key = list(data.keys())[0]
    seed = list(data[key]["data"].keys())[0]
    model_state = data[key]["data"][seed]["model"]

    ds_config = {
        "type": "WAYEEGGALDataset-Low",
        "batch_size": None,
        "shuffle": False,
        "subjects": subjects[i],
        # "subjects": [2, 3],  # seed 1
        "percent": percent,

    }

    ds = dataset.WAYEEGGALDataset("low", ds_config, 1)
    dl = loader.NumpyLoader(ds, batch_size, ds_config["shuffle"])

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
