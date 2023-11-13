import chex
from functools import partial
import numpy as np
import jaxpruner
import yaml
from clu import metrics
from flax import struct
from flax.training import orbax_utils
import optax
import jax.numpy as jnp
import jax
from jax import jit
import time
import sklearn
import pickle
import hashlib
import os
from pprint import pprint

import sparseeg.data.dataset as dataset
import sparseeg.data.loader as loader
import sparseeg.training.state as training_state
from sparseeg.training.ttv_split import TTVSplitTrainer
import sparseeg.util.construct as construct

import flax.linen as nn
import torch
from math import gcd
import warnings


@struct.dataclass
class WeightedSoftmaxCrossEntropyWithIntegerLabels:
    label_weights: jnp.ndarray

    @classmethod
    def weights(cls, train_ds):
        n_classes = train_ds.n_classes
        y_samples = train_ds.y_samples
        n_samples = len(y_samples)

        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        return n_samples / (n_classes * jnp.bincount(y_samples))

    def compute(
        self,
        logits: chex.Array,
        labels: chex.Array,
    ) -> chex.Array:
        chex.assert_type([logits], float)
        chex.assert_type([labels], int)

        weights = self.label_weights[labels]

        # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except
        # that we avoid subtracting the normalizer from all values, just from
        # the values for the correct labels.
        logits_max = jnp.max(logits, axis=-1, keepdims=True)
        logits -= jax.lax.stop_gradient(logits_max)
        label_logits = jnp.take_along_axis(
            logits, labels[..., None], axis=-1,
        )[..., 0]
        log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
        return weights * (log_normalizers - label_logits)


def get_data(identifier, config, seed):
    return dataset.load(identifier, config, seed)


####################################################################
# Adapted from
#   - https://flax.readthedocs.io/en/latest/getting_started.html
#   - https://github.com/google/CommonLoopUtils/blob/main/clu/metrics.py
####################################################################
@struct.dataclass
class Metrics(training_state.MetricsCollection):
    accuracy: metrics.Accuracy
    loss_mean: metrics.Average.from_output("loss")
    loss_std: metrics.Std.from_output("loss")


@partial(jit, static_argnames=("loss_fn"))
def compute_metrics(*, state, batch, loss_fn):
    logits = state.apply_fn({'params': state.params}, batch['inputs'])
    # loss = optax.softmax_cross_entropy_with_integer_labels(
    #     logits=logits, labels=batch['labels']
    # ).mean()
    loss = loss_fn(
        logits=logits, labels=batch['labels']
    ).mean()
    labels = jnp.array(batch["labels"], dtype=jnp.int32)
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss,
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state
####################################################################


def default_config():
    return {
        'dataset': {
            'batch_size': 10,
            'shuffle': True,
            'type': 'ClassifierDataset-Default',
        },
        'model': {
            'type': 'DenseMLP',
            'activations': ['relu', 'relu'],
            'hidden_layers': [64, 64],
            'optim': {
                'args': [],
                'kwargs': {'learning_rate': 0.01},
                'type': 'adam',
            },
        },
        'seed': 1,
        'epochs': 10,
    }


def working_experiment():
    return main_experiment(default_config(), verbose=True)


def main_experiment(config, save_file, verbose=False):
    seed = config["seed"]
    torch.manual_seed(seed)  # Needed to seed the shuffling procedure
    np.random.seed(seed)

    epochs = config["epochs"]

    weighted_loss = config["weighted_loss"]

    train_percent = config["train_percent"]
    valid_percent = config["valid_percent"]

    dataset_config = config["dataset"]
    batch_size = dataset_config["batch_size"]
    shuffle = dataset_config["shuffle"]

    # Get model parameters
    model_config = config["model"]
    model_type = model_config["type"]
    hidden_layers = model_config["hidden_layers"]
    activations = model_config["activations"]

    # Get optimiser parameters
    optim_config = model_config["optim"]
    opt_type = optim_config["type"]

    dataset_name = dataset_config["type"]

    trainer = TTVSplitTrainer(
        experiment_loop, config, model_fn, optim_fn, dataset_fn, batch_size,
        shuffle, dataset.StratifiedTTV, train_percent, valid_percent,

    )

    data = trainer.run(seed, epochs, weighted_loss, verbose)
    return {"data": data, "config": config}


def model_fn(model_config, seed, train_ds):
    model_type = model_config["type"]
    hidden_layers = model_config["hidden_layers"]
    activations = model_config["activations"]
    return construct.model(
        model_type, seed, train_ds, hidden_layers, activations,
        jax.nn.initializers.glorot_normal(),  # TODO: get from config
    )


def optim_fn(optim_config):
    optim_type = optim_config["type"]
    return construct.optim(optim_type, optim_config)


def dataset_fn(dataset_config, seed):
    dataset_name = dataset_config["type"]
    return get_data(dataset_name, dataset_config, seed)


def record_metrics(
    type_, state, full_dataset, datasets_for_labels, data, loss_fn,
):
    ####################################################################
    # Compute metrics for each separate label
    ####################################################################
    for label in range(len(datasets_for_labels)):
        ds = datasets_for_labels[label]
        if len(ds) == 0:
            continue

        dl = loader.NumpyLoader(ds, batch_size=len(ds), shuffle=False)

        # Compute label metrics
        for x_batch, y_batch in dl:
            batch = {"inputs": x_batch, "labels": y_batch}
            state = compute_metrics(
                state=state, batch=batch, loss_fn=loss_fn,
            )
            for metric, value in state.metrics.compute().items():
                key = f"{type_}_{metric}"
                data[key]["label-by-label"][label].append(value.item())

            state = state.replace(metrics=state.metrics.empty())
    ####################################################################

    ####################################################################
    # Compute metrics for all labels/combined dataset
    ####################################################################
    dl = loader.NumpyLoader(
        full_dataset, batch_size=len(full_dataset), shuffle=False,
    )

    # Compute label metrics
    for x_batch, y_batch in dl:
        batch = {"inputs": x_batch, "labels": y_batch}
        state = compute_metrics(
            state=state, batch=batch, loss_fn=loss_fn,
        )
        for metric, value in state.metrics.compute().items():
            key = f"{type_}_{metric}"
            data[key]["combined"].append(value.item())

        state = state.replace(metrics=state.metrics.empty())

    return state.replace(metrics=state.metrics.empty())


# To cache on an epoch-by-epoch basis, what we could do is to cache the data
# generated from this function somewhere else. When the function starts, it
# first loads that data (and current epoch number).
#
# I don't think we need to pickle the dataset and data loaders, since we are
# only caching on a fold-by-fold basis, and we can easily re-get these folds
def experiment_loop(
    cv, seed, epochs, model, optim, train_ds, train_dl, test_ds,
    valid_ds, weighted_loss=True, verbose=False,
):
    assert train_ds.n_classes == test_ds.n_classes
    assert train_ds.n_classes == valid_ds.n_classes
    start_time = time.time()

    # Construct the training state
    init_rng = jax.random.key(seed)
    state = training_state.create(
        model, init_rng, Metrics, train_ds[0][0], optim,
    )
    del init_rng

    # TODO: use this loss function when computing metrics and when stepping the
    # model! Everywhere we use optax.softmax_cross_entropy_with_integer_labels,
    # we should replace with a call to a function argument where the loss is
    # passed in. That way, we can pass in
    # optax.softmax_cross_entropy_with_integer_labels or this weighted loss
    # class/function instead!
    if weighted_loss:
        weights = WeightedSoftmaxCrossEntropyWithIntegerLabels.weights(
            train_ds,
        )
        loss = WeightedSoftmaxCrossEntropyWithIntegerLabels(
            weights,
        ).compute
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels

    data = {}
    for type_ in ("train", "test", "valid"):
        for metric in state.metrics.keys():
            key = f"{type_}_{metric}"
            data[key] = {}
            data[key]["label-by-label"] = [
                [] for _ in range(train_ds.n_classes)
            ]
            data[key]["combined"] = []

    train_datasets_for_labels = tuple(
        train_ds.get_dataset_for(label)
        for label in train_ds.classes
    )
    test_datasets_for_labels = tuple(
        test_ds.get_dataset_for(label)
        for label in test_ds.classes
    )
    valid_datasets_for_labels = tuple(
        valid_ds.get_dataset_for(label)
        for label in valid_ds.classes
    )

    # Record performance before training
    state = record_metrics(
        "train", state, train_ds, train_datasets_for_labels, data,
        loss,
    )
    state = record_metrics(
        "test", state, test_ds, test_datasets_for_labels, data,
        loss,
    )
    state = record_metrics(
        "valid", state, valid_ds, valid_datasets_for_labels, data,
        loss,
    )

    for epoch in range(epochs):
        _start_time = time.time()
        print(f"epoch {epoch} started: {_start_time}")
        # Train for one epoch
        for x_batch, y_batch in train_dl:
            train_batch = {"inputs": x_batch, "labels": y_batch}

            state = training_state.step(
                state,
                train_batch,
                loss,
            )

            post_params = state.update_sparsity()
            state = state.replace(params=post_params)

        # Record performance at the end of each epoch
        state = record_metrics(
            "train", state, train_ds, train_datasets_for_labels, data,
            loss,
        )
        state = record_metrics(
            "test", state, test_ds, test_datasets_for_labels, data,
            loss,
        )
        state = record_metrics(
            "valid", state, valid_ds, valid_datasets_for_labels, data,
            loss,
        )
        print(f"epoch {epoch} ended: {time.time() - _start_time}")

    data["total_time"] = time.time() - start_time
    data["model"] = state

#     # print(data["train_accuracy"])
#     # print(data["valid_accuracy"])
#     # print()
#     # for i in range(4):
#     #     print(data["train_accuracy"][i][-3:])
#     #     print(data["valid_accuracy"][i][-3:])
#     #     print()
#     print()
#     print()
#     print()
#     print(data["valid_confusion"]["label-by-label"][1])
#     print(len(data["valid_confusion"]["combined"]))
#     print()

    return data
