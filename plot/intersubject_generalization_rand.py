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

for seed in range(30):
    dl = loader.NumpyLoader(ds, len(ds), ds_config["shuffle"])
    # Compute label metrics
    labels = []
    predictions = []
    for x_batch, y_batch in dl:
        rng = np.random.default_rng(seed=seed)
        # pred = rng.integers(0, 4, y_batch.shape[0])
        print(y_batch, y_batch.shape)
        pred = rng.choice(y_batch, size=len(ds))
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

print("Accuracy")
acc = np.array(acc)
significance = 0.05
conf = bs.bootstrap(acc, stat_func=bs_stats.mean, alpha=significance)
print(f"{acc.mean():.3f} ({conf.lower_bound:.3f}, {conf.upper_bound:.3f})")

print("Precision")
print(pr.shape)
for c in range(4):
    significance = 0.05
    conf = bs.bootstrap(pr[:, c], stat_func=bs_stats.mean, alpha=significance)
    print(
        f"{pr.mean():.3f}, ({conf.lower_bound:.3f}, {conf.upper_bound:.3f})"
    )
    print()

print("Recall")
for c in range(4):
    significance = 0.05
    conf = bs.bootstrap(r[:, c], stat_func=bs_stats.mean, alpha=significance)
    print(
        f"{r.mean():.3f}, ({conf.lower_bound:.3f}, {conf.upper_bound:.3f})"
    )
    print()
    print(r.mean())
