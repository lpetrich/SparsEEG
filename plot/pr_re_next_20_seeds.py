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

data_file = "results/next_20_seeds/weight_pruning_500epochs_1subject_weighted_next_20_seeds/combined"
chptr = orbax.checkpoint.PyTreeCheckpointer()
data = chptr.restore(data_file)


ds_config = {
    "type": "WAYEEGGALDataset-Low",
    "batch_size": None,
    "shuffle": False,
    "n_subjects": 0,
    "empty": True,
    "percent": 1.0,

}

ds = dataset.WAYEEGGALDataset("low", ds_config, 1)

pr, re = [], []
seed_data = data["None"]["data"]
for i, seed in enumerate(seed_data):
    _ds = deepcopy(ds)
    ds_data = seed_data[seed]["dataset"]["test"]
    _ds.data = ds_data
    dl = loader.NumpyLoader(_ds, len(_ds), False)

    # model_state = data["None"]["data"][seed]["model"]
    # model = default.model_fn(data["None"]["config"]["model"], 1, _ds)

    labels = []
    predictions = []
    for x_batch, y_batch in dl:
        batch = {"inputs": x_batch, "labels": y_batch}

        # logits = model.apply(
        #     {'params': model_state["params"]}, batch['inputs'],
        # )
        # pred = jnp.argmax(logits, axis=-1)

        rng = np.random.default_rng(seed=int(i))
        pred = rng.integers(0, 4, size=y_batch.shape[0])
        predictions.extend(pred)
        labels.extend(y_batch)

    _pr, _re = skmetrics.precision_recall_fscore_support(
        labels, predictions,
    )[0:2]
    pr.append(_pr)
    re.append(_re)

pr = np.array(pr)
re = np.array(re)

significance = 0.05
ci = [[], []]
for j in range(pr.shape[1]):
    conf = bs.bootstrap(
        pr[:, j], stat_func=bs_stats.mean,
        alpha=significance,
    )
    ci[0].append(conf.lower_bound)
    ci[1].append(conf.upper_bound)
ci = np.array(ci)

print(pr.mean(axis=0))
print(
    f"({ci[0][0]:.3f}, {ci[1][0]:.3f}) & ({ci[0][1]:.3f}, {ci[1][1]:.3f}) & " +
    f"({ci[0][2]:.3f}, {ci[1][2]:.3f}) & ({ci[0][3]:.3f}, {ci[1][3]:.3f})"
)

ci = [[], []]
for j in range(re.shape[1]):
    conf = bs.bootstrap(
        re[:, j], stat_func=bs_stats.mean,
        alpha=significance,
    )
    ci[0].append(conf.lower_bound)
    ci[1].append(conf.upper_bound)
ci = np.array(ci)
print(
    f"({ci[0][0]:.3f}, {ci[1][0]:.3f}) & ({ci[0][1]:.3f}, {ci[1][1]:.3f}) & " +
    f"({ci[0][2]:.3f}, {ci[1][2]:.3f}) & ({ci[0][3]:.3f}, {ci[1][3]:.3f})"
)
print(re.mean(axis=0))
