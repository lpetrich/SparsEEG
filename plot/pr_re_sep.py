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
import jax
import jax.numpy as jnp
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

algs = {"dense": [[], []], "set": [[], []], "wp": [[], []]}
for alg in ("dense", "set", "wp"):
    for p in train_percent:
        p_pr_data = []
        p_r_data = []
        for seed in range(3, 18):
            fname = f"plot_data/{alg}_{str(seed)}_{str(p)}.pkl"
            with open(fname, "rb") as infile:
                d = pickle.load(infile)

            p_pr_data.append(d[0])
            p_r_data.append(d[1])

        algs[alg][0].append(p_pr_data)
        algs[alg][1].append(p_r_data)

algs["dense"][0] = np.array(algs["dense"][0])
algs["dense"][1] = np.array(algs["dense"][1])
algs["set"][0] = np.array(algs["set"][0])
algs["set"][1] = np.array(algs["set"][1])
algs["wp"][0] = np.array(algs["wp"][0])
algs["wp"][1] = np.array(algs["wp"][1])

n_classes = 4
fig = plt.figure(figsize=(15, 10))
axes = fig.subplots(2, n_classes)

significance = 0.1

colours = ["black", "red", "blue"]
# labels = ["Dense", "SET", "Weight Pruning"]
# for i, alg in enumerate(("dense", "set", "wp")):
#     for c in range(n_classes):
#         ax = axes[0, c]
#         pr = algs[alg][0][:, :, c]
#         pr_mean = pr.mean(axis=1)

#         ci = [[], []]
#         for j in range(pr.shape[0]):
#             conf = bs.bootstrap(
#                 pr[j, :], stat_func=bs_stats.mean,
#                 alpha=significance,
#             )
#             ci[0].append(conf.lower_bound)
#             ci[1].append(conf.upper_bound)
#         ci = np.array(ci)

#         ax.plot(train_percent, pr_mean, label=labels[i], color=colours[i])
#         ax.fill_between(
#             train_percent, ci[0], ci[1], color=colours[i], alpha=0.2,
#         )

#         ax.set_ylim(0.8, 1.0)
#         if c == 0:
#             ax.set_ylabel("Precision", fontsize=16)
#             ax.set_yticks([0.8, 0.9, 1.0], labels=[0.8, 0.9, 1.0], fontsize=12)
#             ax.legend(loc="lower right", fontsize=12)
#         else:
#             ax.set_yticks([])

#         ax.set_xticks([])
#         ax.set_title(f"Class {c}")

# for i, alg in enumerate(("dense", "set", "wp")):
#     for c in range(n_classes):
#         ax = axes[1, c]
#         r = algs[alg][1][:, :, c]
#         r_mean = r.mean(axis=1)

#         ci = [[], []]
#         for j in range(r.shape[0]):
#             conf = bs.bootstrap(
#                 r[j, :], stat_func=bs_stats.mean,
#                 alpha=significance,
#             )
#             ci[0].append(conf.lower_bound)
#             ci[1].append(conf.upper_bound)
#         ci = np.array(ci)

#         ax.plot(train_percent, r_mean, label=labels[i], color=colours[i])
#         ax.fill_between(
#             train_percent, ci[0], ci[1], color=colours[i], alpha=0.2,
#         )

#         ax.set_ylim(0.8, 1.0)
#         if c == 0:
#             ax.set_ylabel("Recall", fontsize=16)
#             ax.set_yticks([0.8, 0.9, 1.0], labels=[0.8, 0.9, 1.0], fontsize=12)
#             ax.text(1.33, 0.77, "Percentage of Data Trained on", fontsize=16)
#         else:
#             ax.set_yticks([])

#         ax.set_xlabel("", fontsize=16)
#         ax.set_xticks([0.1, 0.45, 0.8], labels=[0.1, 0.45, 0.8], fontsize=12)

# fig.savefig("/home/samuel/test.png", bbox_inches="tight")
# fig.savefig("/home/samuel/test.svg", bbox_inches="tight")

for alg in ("dense", "set", "wp"):
    for precision in (True, False):
        colours = {"dense": "black", "set": "red", "wp": "blue"}
        c = 1
        p = 2
        fig = plt.figure(figsize=(15, 10))
        axes = fig.subplots(4, 6)
        suptitle = (
            "Distribution of " + ("Precision" if precision else "Recall") +
            " for "
        )
        if alg == "dense":
            suptitle += "Dense"
        else:
            suptitle += alg.upper()
        fig.suptitle(suptitle, fontsize=32)

        for c in range(4):
            for i, p in enumerate(range(6)):
                ax = axes[c, p]
                # ax.hist(algs[alg][0][p, :, c])
                sns.histplot(
                    algs[alg][0 if precision else 1][p, :, c], ax=ax, kde=True,
                    color=colours[alg],
                )
                if i != 0:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(f"Class {c}", fontsize=16)

                if c != 3:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(
                        "Precision" if precision else "Recall",
                        fontsize=16,
                    )

                if c == 0:
                    tp = str(int(train_percent[p] * 100))
                    ax.set_title(f"{tp}%", fontsize=16)

        label = "precision" if precision else "recall"

        fig.savefig(f"/home/samuel/{alg}_histogram_{label}.png", bbox_inches="tight")
        fig.savefig(f"/home/samuel/{alg}_histogram_{label}.svg", bbox_inches="tight")
