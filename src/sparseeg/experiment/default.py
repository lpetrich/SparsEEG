from functools import partial
import numpy as np
import jaxpruner
import yaml
from clu import metrics
from flax import struct
import optax
import jax.numpy as jnp
import jax
from jax import jit
import time

import sparseeg.data.dataset as dataset
import sparseeg.data.loader as loader
import sparseeg.training.state as training_state
from sparseeg.training.cross_validation import NestedCrossValidation
from sparseeg.training.cross_validation import adjust_batch_size
import sparseeg.util.construct as construct

import flax.linen as nn
import torch
from math import gcd
import warnings
from pprint import pprint


def get_data(identifier, seed):
    return dataset.load(identifier, seed)


####################################################################
# Adapted from https://flax.readthedocs.io/en/latest/getting_started.html
####################################################################
@struct.dataclass
class Metrics(training_state.MetricsCollection):
    accuracy: metrics.Accuracy
    loss_mean: metrics.Average.from_output("loss")
    loss_std: metrics.Std.from_output("loss")


@jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['inputs'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
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


def main_experiment(config, verbose=False):
    seed = config["seed"]
    torch.manual_seed(seed)  # Needed to seed the shuffling procedure

    epochs = config["epochs"]

    n_external_folds = config["n_external_folds"]
    n_internal_folds = config["n_internal_folds"]

    dataset_config = config["dataset"]
    external_batch_size = dataset_config["external_batch_size"]
    internal_batch_size = dataset_config["internal_batch_size"]
    shuffle_external = dataset_config["shuffle_external"]
    shuffle_internal = dataset_config["shuffle_internal"]

    # Get model parameters
    model_config = config["model"]
    model_type = model_config["type"]
    hidden_layers = model_config["hidden_layers"]
    activations = model_config["activations"]

    # Get optimiser parameters
    optim_config = model_config["optim"]
    opt_type = optim_config["type"]

    def model_fn(seed, train_ds):
        return construct.model(
            model_type, seed, train_ds, hidden_layers, activations,
            jax.nn.initializers.glorot_normal(),  # TODO: get from config
        )

    def optim_fn():
        return construct.optim(opt_type, optim_config)

    dataset_name = dataset_config["type"]

    def dataset_fn(seed):
        return get_data(dataset_name, seed)

    cv = NestedCrossValidation(
        experiment_loop, model_fn, optim_fn, n_external_folds,
        n_internal_folds, dataset_fn, external_batch_size, internal_batch_size,
        shuffle_external, shuffle_internal, dataset.StratifiedKFold
    )

    cv_data = cv.run(seed, epochs, verbose)
    return {"data": cv_data, "config": config}


def record_metrics(type_, state, datasets_for_labels, data):
    for label in range(len(datasets_for_labels)):
        ds = datasets_for_labels[label]
        dl = loader.NumpyLoader(ds, batch_size=len(ds), shuffle=False)

        # Compute label metrics
        for x_batch, y_batch in dl:
            batch = {"inputs": x_batch, "labels": y_batch}
            state = compute_metrics(state=state, batch=batch)
            for metric, value in state.metrics.compute().items():
                key = f"{type_}_{metric}"
                data[key][label].append(value.item())

            state = state.replace(metrics=state.metrics.empty())

    return state.replace(metrics=state.metrics.empty())


def experiment_loop(
    seed, epochs, model, optim, train_ds, train_dl, test_ds, test_dl,
    verbose=False,
):
    assert train_ds.n_classes == test_ds.n_classes

    # Construct the training state
    init_rng = jax.random.key(seed)
    state = training_state.create(
        model, init_rng, Metrics, train_ds[0][0], optim,
    )
    del init_rng

    data = {}
    for type_ in ("train", "test"):
        for metric in state.metrics.keys():
            key = f"{type_}_{metric}"
            data[key] = [[] for _ in range(train_ds.n_classes)]

    train_datasets_for_labels = tuple(
        train_ds.get_dataset_for(label)
        for label in range(train_ds.n_classes)
    )
    test_datasets_for_labels = tuple(
        test_ds.get_dataset_for(label)
        for label in range(test_ds.n_classes)
    )

    # Record performance before training
    state = record_metrics("train", state, train_datasets_for_labels, data)
    state = record_metrics("test", state, test_datasets_for_labels, data)

    start_time = time.time()
    for epoch in range(epochs):
        # Train for one epoch
        for x_batch, y_batch in train_dl:
            train_batch = {"inputs": x_batch, "labels": y_batch}

            state = training_state.step(state, train_batch)

            # TODO: the train/test accuracy doesn't seem to change when
            # updating sparsity every step. When updating sparsity every N
            # steps for N > 1, then learning actually seems to occur
            #
            # Looks like the same accuracy as the first epoch on the dense
            # network, meaning that it looks like no updating is happening??
            post_params = state.update_sparsity()
            state = state.replace(params=post_params)

        # Record performance at the end of each epoch
        state = record_metrics("train", state, train_datasets_for_labels, data)
        state = record_metrics("test", state, test_datasets_for_labels, data)

    data["total_time"] = time.time() - start_time
    return data
