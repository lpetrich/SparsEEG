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


# data_file = f"./results/single_subject_dense_500epochs_weighted_next_10_seeds/model_params/"
# data_file = f"./results/single_subject_set_500epochs_weighted_next_10_seeds/combined_valid_accuracy/"
# data_file = f"./results/single_subject_weight_pruning_500epochs_weighted_next_10_seeds/combined_valid_accuracy/"
# data_file = f"./results/single_subject_set_500epochs_weighted_next_5_seeds_/hyper_tuning_combined/"
data_file = f"./results/single_subject_weight_pruning_500epochs_weighted_next_5_seeds_/hyper_tuning_combined/"
chptr = orbax.checkpoint.PyTreeCheckpointer()

print("Loading...")
data = chptr.restore(data_file)
print("Loaded")

train_percent = get_train_percents(data)
train_percent = sorted(train_percent)
seeds = get_seeds(data)

to_tune = "valid_accuracy"

# TODO: maybe report perf with and without this? I bet SET will be >> Dense
# with this set to True
samplewise = False

d = []
p = train_percent[-1]
pr_dists = []
r_dists = []

hypers = []

for p in train_percent:
    new_data = hyper.satisfies(
        data,
        lambda x: (
            x["train_percent"] == p and
            x["dataset"]["batch_size"] == 8192
        ),
    )

    perfs = hyper.perfs(
        new_data, to_tune, inner_agg, outer_agg, combined=False,
    )
    h = hyper.best(perfs, np.mean)
    # print("=== === === ===")
    # print("Hyper setting ind for new_data:", h, "train %:", p)
    # print("=== === === ===")
    # print(h, p)
    # pprint(new_data[str(h)]["config"])
    # print()
    # print()
    # continue
    hypers.append(new_data[str(h)]["config"])
    continue

    seed_d = []
    seed_conf = []
    for seed in seeds:
        seed_key = f"seed_{seed}"
        ds_data = new_data[str(h)]["data"][seed_key]["dataset"]["valid"]
        ds_config = {
            "type": "WAYEEGGALDataset-low",
            "n_subjects": 1,
            "empty": True,
        }
        ds = dataset.load(ds_config["type"], ds_config, seed)
        ds.data = ds_data

        dl = loader.NumpyLoader(ds, len(ds), False)

        model = default.model_fn(new_data[str(h)]["config"]["model"], seed, ds)
        model_state = new_data[str(h)]["data"][seed_key]["model"] #data --> new

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

        # print(np.unique(labels))
        # print(np.unique(predictions))
        pr, r, f, _ = (
            skmetrics.precision_recall_fscore_support(labels, predictions)
        )
        seed_d.append([pr, r, f])
        seed_conf.append(skmetrics.multilabel_confusion_matrix(
            labels, predictions, samplewise=samplewise,
        ))

        print(seed_conf[-1])
        print(seed_conf[-1].shape)

    # d = np.array(seed_d).mean(axis=0) # Mean along seeds

    # Get TP, TN, FP, FN
    conf = np.array(seed_conf) # (10, 4, 2, 2)
    conf = conf.mean(axis=0)
    TN = conf[:, 0, 0]
    FN = conf[:, 1, 0]
    TP = conf[:, 1, 1]
    FP = conf[:, 0, 1]

    prior_λ = 0.5

    # Use this to compute CIs or credible intervals:
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.64.142&rep=rep1&type=pdf
    # Also use it to compute CIs for the other experiments, with multiple subjects
    # test data and generalizing to new subjects

    # Credible intervals
    posterior_dist_pr = scipy.stats.beta(TP + prior_λ, FP + prior_λ)
    posterior_dist_r = scipy.stats.beta(TP + prior_λ, FN + prior_λ)

    pr_dists.append(posterior_dist_pr)
    r_dists.append(posterior_dist_r)

with open("./plot/wp_hypers_combined_5_seeds.json", "w") as outfile:
    json.dump(hypers, outfile)

exit()

fig = plt.figure(figsize=(10, 5))
axes = fig.subplots(1, 2)

significance = 0.01
pr_means = np.array([pr.mean() for pr in pr_dists])
pr_credible_low = np.array([pr.ppf(significance / 2) for pr in pr_dists])
pr_credible_high = np.array(
    [pr.ppf(1 - (significance / 2)) for pr in pr_dists],
)
print(pr_means.shape)
print(pr_credible_low.shape)
print(pr_credible_high.shape)

ax = axes.ravel()[0]
ax.set_title("Precision")
colours = ["black", "red", "blue", "green"]
labels = [f"Class {n}" for n in range(pr_means.shape[-1])]
for i in range(pr_means.shape[-1]):
    ax.plot(
        train_percent, pr_means[:, i], color=colours[i], label=labels[i],
    )
    ax.fill_between(
        train_percent,
        pr_credible_low[:, i],
        pr_credible_high[:, i],
        alpha=0.2,
        color=colours[i],
    )
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.7, 0.85, 1.0])
    ax.legend(loc="lower right")
    ax.set_ylabel("Precision/Recall")
    ax.set_xlabel("Percentage of Data used for Training")

r_means = np.array([r.mean() for r in r_dists])
r_credible_low = np.array([r.ppf(significance / 2) for r in r_dists])
r_credible_high = np.array(
    [r.ppf(1 - (significance / 2)) for r in r_dists],
)
print(r_means.shape)
print(r_credible_low.shape)
print(r_credible_high.shape)

ax = axes.ravel()[1]
ax.set_title("Recall")
for i in range(r_means.shape[-1]):
    ax.plot(
        train_percent, r_means[:, i], color=colours[i], label=labels[i],
    )
    ax.fill_between(
        train_percent,
        r_credible_low[:, i],
        r_credible_high[:, i],
        alpha=0.2,
        color=colours[i],
    )
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xlabel("Percentage of Data used for Training")

fig.savefig("/home/samuel/test.png", bbox_inches="tight")

# print("Precision Credible Interval")
# print(posterior_dist_pr.ppf(0.975))
# print(posterior_dist_pr.ppf(0.025))
# print(posterior_dist_pr.mean())

# print("Recall Credible Interval")
# print(posterior_dist_r.ppf(0.975))
# print(posterior_dist_r.ppf(0.025))
# print(posterior_dist_r.mean())


# https://stackoverflow.com/questions/57482356/generating-confidence-interval
# -for-precision-recall-curve
