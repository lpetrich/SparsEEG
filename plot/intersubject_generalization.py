import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
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


def inner_agg(x):
    return np.mean(x)


def outer_agg(x, axis):
    return np.mean(x, axis=axis)


alg = "weight_pruning"
i = 9

# We'll need to average over seeds eventually
data_file = f"./results/next_20_seeds/{alg}_500epochs_{i}subject_weighted_next_20_seeds/combined/"
chptr = orbax.checkpoint.PyTreeCheckpointer()

print("Loading...")
data = chptr.restore(data_file)
print("Loaded")

print("Tuning")
to_tune = "valid_accuracy"
# perfs = hyper.perfs(data, to_tune, inner_agg, outer_agg, combined=False)
# print(perfs)
# key = str(hyper.best(perfs, np.mean))
key = "None"
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

pr = []
r = []
acc = []

for seed in data[key]["data"].keys():
    model_state = data[key]["data"][seed]["model"]

    batch_size = len(ds)
    dl = loader.NumpyLoader(ds, batch_size, ds_config["shuffle"])

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

        labels.extend(y_batch)
        predictions.extend(pred)

    _pr, _r = skmetrics.precision_recall_fscore_support(
        labels, predictions
    )[0:2]
    pr.append(_pr)
    r.append(_r)

    # TODO: Ci
    correct = np.array(labels) == np.array(predictions)
    _acc = sum(correct) / len(labels)
    acc.append(_acc)

pr = np.array(pr)
r = np.array(r)

# print(pr.shape)

acc = np.array(acc)
significance = 0.05
conf = bs.bootstrap(acc, stat_func=bs_stats.mean, alpha=significance)
print(f"{acc.mean():.3f} ({conf.lower_bound:.3f}, {conf.upper_bound:.3f})")

pr_ci = [[], []]
for j in range(pr.shape[1]):
    conf = bs.bootstrap(
        pr[:, j], stat_func=bs_stats.mean,
        alpha=significance,
    )
    pr_ci[0].append(conf.lower_bound)
    pr_ci[1].append(conf.upper_bound)
ci = pr_ci
print(
    f"({ci[0][0]:.3f}, {ci[1][0]:.3f}) & ({ci[0][1]:.3f}, {ci[1][1]:.3f}) & " +
    f"({ci[0][2]:.3f}, {ci[1][2]:.3f}) & ({ci[0][3]:.3f}, {ci[1][3]:.3f})"
)

r_ci = [[], []]
for j in range(pr.shape[1]):
    conf = bs.bootstrap(
        r[:, j], stat_func=bs_stats.mean,
        alpha=significance,
    )
    r_ci[0].append(conf.lower_bound)
    r_ci[1].append(conf.upper_bound)
ci = r_ci
print(
    f"({ci[0][0]:.3f}, {ci[1][0]:.3f}) & ({ci[0][1]:.3f}, {ci[1][1]:.3f}) & " +
    f"({ci[0][2]:.3f}, {ci[1][2]:.3f}) & ({ci[0][3]:.3f}, {ci[1][3]:.3f})"
)

# labels = np.array(labels).ravel()
# predictions = np.array(predictions).ravel()

# print(type(labels), labels.shape)
# print(type(predictions), predictions.shape)

# pp_matrix_from_data(labels, predictions, cmap="BuPu")
# print(skmetrics.precision_recall_fscore_support(
#     labels, predictions
# ))
