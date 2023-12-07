import pandas as pd
from copy import deepcopy
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
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def inner_agg(x):
    return np.mean(x)


def outer_agg(x, axis):
    return np.mean(x, axis=axis)


# We'll need to average over seeds eventually
# data_file = "results/next_20_seeds/set_500epochs_9subject_weighted_next_20_seeds/combined"
data_file = "./results/set_500epochs_9subject_weighted/model_params/"
chptr = orbax.checkpoint.PyTreeCheckpointer()

print("Loading...")
data = chptr.restore(data_file)
# data["0"] = data["None"]
# del data["None"]
print("Loaded")

print("Tuning")
to_tune = "valid_accuracy"
perfs = hyper.perfs(data, to_tune, inner_agg, outer_agg, combined=False)
print(perfs)
key = str(hyper.best(perfs, np.mean))
pprint(data[key]["config"])
print("Best:", key)

ds_config = {
    "type": "WAYEEGGALDataset-Low",
    "batch_size": None,
    "shuffle": False,
    "subjects": [11, 12],  # seed 1
    "percent": 1.0,

}
ds = dataset.WAYEEGGALDataset("low", ds_config, 1)
batch_size = len(ds)

pr = []
r = []

for seed in data[key]["data"].keys():
    model_state = data[key]["data"][seed]["model"]

    dl = loader.NumpyLoader(deepcopy(ds), batch_size, ds_config["shuffle"])

    model = default.model_fn(data[key]["config"]["model"], 1, ds)

    # Compute label metrics
    labels = []
    predictions = []
    for x_batch, y_batch in dl:
        batch = {"inputs": x_batch, "labels": y_batch}
        logits = model.apply(
            {'params': model_state["params"]}, batch['inputs'],
        )

        pred = jnp.argmax(logits, axis=-1)
        predictions.extend(pred)
        labels.extend(y_batch)

    _pr, _r = skmetrics.precision_recall_fscore_support(
        labels, predictions,
    )[0:2]
    pr.append(_pr)
    r.append(_r)

pr = np.array(pr)
r = np.array(r)

print(pr.shape)
print(r.shape)

ci = [[], []]
for j in range(pr.shape[0]):
    conf = bs.bootstrap(
        pr[j, i, :], stat_func=bs_stats.mean,
        alpha=significance,
    )
    ci[0].append(conf.lower_bound)
    ci[1].append(conf.upper_bound)
ci = np.array(ci)
ax.fill_between(x_values[:, i], ci[0], ci[1], alpha=0.2)


print(type(labels), labels.shape)
print(type(predictions), predictions.shape)

pp_matrix_from_data(labels, predictions, cmap="BuPu")
