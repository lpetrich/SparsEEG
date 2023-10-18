# Eventually, this will be made into an experiment in the experiments/
# direcotory, and we can use Andy's code to schedule
import numpy as np
import yaml
from clu import metrics
from flax import struct
import optax
import jax.numpy as jnp
import jax
from jax import jit
# import src.project.data.dataset as dataset
# import src.project.data.loader as loader
# import src.project.training.state as training_state
# from src.project.training.cross_validation import NestedCrossValidation
# from src.project.training.cross_validation import adjust_batch_size
# import src.project.util.construct as construct
import data.dataset as dataset
import data.loader as loader
import training.state as training_state
from training.cross_validation import NestedCrossValidation
from training.cross_validation import adjust_batch_size
import util.construct as construct

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
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['labels'], loss=loss,
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
    opt_args = optim_config["args"]
    opt_kwargs = optim_config["kwargs"]

    def model_fn(seed, train_ds):
        return construct.model(
            model_type, seed, train_ds, hidden_layers, activations,
            jax.nn.initializers.glorot_normal(), # TODO: get from config
        )

    def optim_fn():
        return construct.optim(opt_type, opt_args, opt_kwargs)

    dataset_name = dataset_config["type"]
    dataset_fn = lambda seed: get_data(dataset_name, seed)

    cv = NestedCrossValidation(
        experiment_loop, model_fn, optim_fn, n_external_folds,
        n_internal_folds, dataset_fn, external_batch_size, internal_batch_size,
        shuffle_external, shuffle_internal, dataset.StratifiedKFold
    )

    return cv.run(seed, epochs, verbose)


def experiment_loop(
    seed, epochs, model, optim, train_ds, train_dl, test_ds, test_dl,
    verbose=False,
):
    # Construct the training state
    init_rng = jax.random.key(seed)
    state = training_state.create(
        model, init_rng, Metrics, train_ds[0][0], optim,
    )
    del init_rng

    metrics_history = {}
    for key in state.metrics.keys():
        metrics_history[f"train_{key}"] = []
        metrics_history[f"test_{key}"] = []

    data = {}
    for type_ in ("train", "test"):
        for metric in state.metrics.keys():
            key = f"{type_}_{metric}"
            data[key] = []

    for epoch in range(epochs):
        # Train for one epoch
        for x_batch, y_batch in train_dl:
            train_batch = {"inputs": x_batch, "labels": y_batch}
            state = training_state.step(state, train_batch)
            state = compute_metrics(state=state, batch=train_batch)

        # Compute training metrics
        for metric, value in state.metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)

        # Reset the training metrics for the next epoch
        state = state.replace(metrics=state.metrics.empty())

        # Compute metrics on test data
        for x_batch, y_batch in test_dl:
            test_batch = {"inputs": x_batch, "labels": y_batch}
            test_state = state
            test_state = compute_metrics(state=test_state, batch=test_batch)

            for metric, value in test_state.metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)

        # Print data at the end of the epoch
        if verbose:
            print(f"==> Epoch {epoch}:")
        for type_ in ("train", "test"):
            for metric in state.metrics.keys():
                key = f"{type_}_{metric}"
                value = metrics_history[key][-1]

                if verbose:
                    print(f"\t{type_.title()} {metric.title()}:\t {value:.3f}")

                data[key].append(value.item())

    return data

