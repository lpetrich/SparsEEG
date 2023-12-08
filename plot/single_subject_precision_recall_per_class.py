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


data_files = [
    "./results/single_subject_dense_500epochs_weighted_next_5_seeds_/" +
    "model_params_combined/",
    "./results/single_subject_set_500epochs_weighted_next_5_seeds_/" +
    "model_params_combined/",
    "./results/single_subject_weight_pruning_500epochs_weighted_next" +
    "_5_seeds_/model_params_combined/",
]
chptr = orbax.checkpoint.PyTreeCheckpointer()


# TODO: maybe report perf with and without this? I bet SET will be >> Dense
# with this set to True
samplewise = False

d = []
pr_dists = []
r_dists = []

pr = []
r = []

if os.path.exists("dists.pkl"):
    with open("dists.pkl", "rb") as infile:
        all_pr, all_r = pickle.load(infile)

    data = chptr.restore(data_files[0])
    train_percent = get_train_percents(data)
    train_percent = sorted(train_percent)
else:
    for data_file in data_files:
        data_pr_dists = []
        data_r_dists = []

        data_pr = []
        data_r = []

        print("Loading...")
        data = chptr.restore(data_file)
        print("Loaded")

        train_percent = get_train_percents(data)
        train_percent = sorted(train_percent)
        seeds = get_seeds(data)

        to_tune = "valid_accuracy"
        for p in train_percent:  # TODO
            new_data = hyper.satisfies(
                data,
                lambda x: (
                    x["train_percent"] == p and
                    x["dataset"]["batch_size"] == 8192
                ),
            )

            # perfs = hyper.perfs(
            #     new_data, to_tune, inner_agg, outer_agg, combined=False,
            # )
            # h = hyper.best(perfs, np.mean)

            if len(new_data.keys()) > 1:
                raise IndexError()
            h = int(list(new_data.keys())[0])

            seed_d = []
            seed_conf = []
            seed_pr = []
            seed_r = []
            for seed in seeds[7:7]:  # TODO
                seed_key = f"seed_{seed}"
                print(str(h))
                ds_data = new_data[str(h)]["data"][seed_key][
                    "dataset"]["test"]
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

                # seed_conf.append(skmetrics.multilabel_confusion_matrix(
                #     labels, predictions, samplewise=samplewise,
                # ))

                p_r_f = skmetrics.precision_recall_fscore_support(
                    labels, predictions,
                )
                seed_pr.append(p_r_f[0])
                seed_r.append(p_r_f[1])

                print("seed_pr:", seed_pr[-1].shape)

            data_pr.append(seed_pr)
            data_r.append(seed_r)

        # pr_dists.append(data_pr_dists)
        # r_dists.append(data_r_dists)

        pr.append(data_pr)
        r.append(data_r)

        with open(f"seed_7.pkl", "wb") as outfile:
            pickle.dump((np.array(pr), np.array(r)), outfile)

    # all_pr_dists = np.array(pr_dists)  # (n_data_files, n_train_percents)
    # all_r_dists = np.array(r_dists)  # (n_data_files, n_train_percents)

    all_pr = np.array(pr)
    all_r = np.array(r)
    print("all:", all_pr.shape)

    with open("dists.pkl", "wb") as outfile:
        pickle.dump((all_pr, all_r), outfile)


n_classes = 4  # all_pr_dists[0, 0].mean().shape[-1]
fig = plt.figure(figsize=(15, 10))
axes = fig.subplots(2, n_classes)

significance = 0.05

colours = ["black", "red", "blue"]
labels = ["Dense", "SET", "Weight Pruning"]
for i in range(len(data_files)):
    for c in range(n_classes):
        ax = axes[0, c]
        pr = all_pr[i, :, :, c]
        pr_mean = pr.mean(axis=1)

        ci = [[], []]
        for j in range(pr.shape[0]):
            conf = bs.bootstrap(
                pr[j, :], stat_func=bs_stats.mean,
                alpha=significance,
            )
            ci[0].append(conf.lower_bound)
            ci[1].append(conf.upper_bound)
        ci = np.array(ci)

        ax.plot(train_percent, pr_mean, label=labels[i], color=colours[i])
        ax.fill_between(
            train_percent, ci[0], ci[1], color=colours[i], alpha=0.2,
        )

        ax.set_ylim(0.8, 1.0)
        if c == 0:
            ax.set_ylabel("Precision", fontsize=16)
            ax.set_yticks([0.8, 0.9, 1.0], labels=[0.8, 0.9, 1.0], fontsize=12)
            ax.legend(loc="lower right", fontsize=12)
        else:
            ax.set_yticks([])

        ax.set_xticks([])
        ax.set_title(f"Class {c}")

for i in range(len(data_files)):
    for c in range(n_classes):
        ax = axes[1, c]
        r = all_r[i, :, :, c]
        r_mean = r.mean(axis=1)

        ci = [[], []]
        for j in range(r.shape[0]):
            conf = bs.bootstrap(
                r[j, :], stat_func=bs_stats.mean,
                alpha=significance,
            )
            ci[0].append(conf.lower_bound)
            ci[1].append(conf.upper_bound)
        ci = np.array(ci)

        ax.plot(train_percent, r_mean, label=labels[i], color=colours[i])
        ax.fill_between(
            train_percent, ci[0], ci[1], color=colours[i], alpha=0.2,
        )

        ax.set_ylim(0.8, 1.0)
        if c == 0:
            ax.set_ylabel("Recall", fontsize=16)
            ax.set_yticks([0.8, 0.9, 1.0], labels=[0.8, 0.9, 1.0], fontsize=12)
            ax.text(1.33, 0.77, "Percentage of Data Trained on", fontsize=16)
        else:
            ax.set_yticks([])

        ax.set_xlabel("", fontsize=16)
        ax.set_xticks([0.1, 0.45, 0.8], labels=[0.1, 0.45, 0.8], fontsize=12)


fig.savefig("/home/samuel/test.png", bbox_inches="tight")
fig.savefig("/home/samuel/test.svg", bbox_inches="tight")

# colours = ["black", "red", "blue"]
# labels = ["Dense", "SET", "Weight Pruning"]
# for i in range(len(data_files)):
#     for c in range(n_classes):
#         ax = axes[0, c]
#         pr_dist = all_pr_dists[i, :]
#         pr_mean = [pr.mean()[c] for pr in pr_dist]
#         pr_credible_low = np.array(
#             [pr.ppf(significance / 2)[c] for pr in pr_dist],
#         )
#         pr_credible_high = np.array(
#             [pr.ppf(1 - (significance / 2))[c] for pr in pr_dist],
#         )
#         ax.plot(
#             train_percent, pr_mean, color=colours[i], label=labels[i],
#         )
#         ax.fill_between(
#             train_percent, pr_credible_low, pr_credible_high,
#             color=colours[i], alpha=0.2,
#         )

#         ax.set_ylim(0.8, 1.0)
#         if c == 0:
#             ax.set_ylabel("Precision", fontsize=16)
#             ax.set_yticks([0.8, 0.9, 1.0])
#             ax.legend()
#         else:
#             ax.set_yticks([])
#             # ax.get_legend().remove()

#         ax.set_xticks([])

# for i in range(len(data_files)):
#     for c in range(n_classes):
#         ax = axes[1, c]
#         r_dist = all_r_dists[i, :]
#         r_mean = [r.mean()[c] for r in r_dist]
#         r_credible_low = np.array(
#             [r.ppf(significance / 2)[c] for r in r_dist],
#         )
#         r_credible_high = np.array(
#             [r.ppf(1 - (significance / 2))[c] for r in r_dist],
#         )
#         ax.plot(train_percent, r_mean, color=colours[i])
#         ax.fill_between(
#             train_percent, r_credible_low, r_credible_high,
#             color=colours[i], alpha=0.2,
#         )

#         ax.set_ylim(0.8, 1.0)
#         if c == 0:
#             ax.set_ylabel("Recall", fontsize=16)
#             ax.set_yticks([0.8, 0.9, 1.0])
#         else:
#             ax.set_yticks([])

#         ax.set_xticks(
#             np.arange(train_percent[0], train_percent[-1] + 0.01, 0.1),
#         )


# fig.savefig("/home/samuel/test.png", bbox_inches="tight")
# fig.savefig("/home/samuel/test.svg", bbox_inches="tight")


# # https://stackoverflow.com/questions/57482356/generating-confidence-interval
# # -for-precision-recall-curve
