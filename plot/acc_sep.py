import seaborn as sns
from flax.training import orbax_utils
from flax.training import train_state
from pprint import pprint
from pretty_confusion_matrix import pp_matrix_from_data
from sparseeg.approximator.mlp import MLP
from sparseeg.experiment import default as default
from sparseeg import dataset
from sparseeg.training import state as training
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import click
import flax.linen as nn
import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import orbax
import os
import pandas as pd
import pickle
import pymc3
import scipy
import sklearn.metrics as skmetrics
import sparseeg.data.dataset as dataset
import sparseeg.data.loader as loader
import sparseeg.util.hyper as hyper
from tqdm import tqdm


train_percent = (0.1, 0.2, 0.3, 0.5, 0.7, 0.8)
colours = {"dense": "black", "set": "red", "wp": "blue"}
title = {"dense": "Dense", "set": "SET", "wp": "Weight Pruning"}

alg = "wp"
data_file = "./results/single_subject_weight_pruning_500epochs_weighted_next_" + \
    "5_seeds_/combined_final"

chptr = orbax.checkpoint.PyTreeCheckpointer()
data = chptr.restore(data_file)

fig = plt.figure(figsize=(20, 4))
axes = fig.subplots(1, 6)

for i, p in enumerate(train_percent):
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

    to_plot = "test_accuracy"
    plot_data = hyper.get(new_data, h, to_plot, combined=True)

    sns.histplot(
        plot_data[-1, :], ax=axes.ravel()[i], kde=True,
        color=colours[alg],
    )

    if i != 0:
        axes.ravel()[i].set_ylabel("")
    else:
        axes.ravel()[i].set_ylabel("Count", fontsize=16)

    axes.ravel()[i].set_xlabel("Accuracy", fontsize=16)
    axes.ravel()[i].set_title(f"{str(int(p * 100))}%", fontsize=16)

fig.suptitle(f"Distribution of Accuracy for {title[alg]}", fontsize=28, y=1.05)

fig.savefig(f"/home/samuel/{alg}_histogram_acc.png", bbox_inches="tight")
fig.savefig(f"/home/samuel/{alg}_histogram_acc.svg", bbox_inches="tight")
