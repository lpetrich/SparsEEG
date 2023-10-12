# Eventually, this will be made into an experiment in the experiments/
# direcotory, and we can use Andy's code to schedule
import pprint
import yaml
from clu import metrics
from flax import struct
import optax
import jax.numpy as jnp
import jax
from jax import jit
import src.project.data.dataset as dataset
import src.project.data.loader as loader
import flax.linen as nn
import src.project.training.state as training_state
import torch
import src.project.util.construct as construct


def get_data(dataset_config, seed):
    train_ds, test_ds = dataset.load(dataset_config["type"], seed)

    torch.manual_seed(seed)  # Needed to seed the shuffling procedure
    batch_size = dataset_config["batch_size"]
    shuffle = dataset_config["shuffle"]
    train_dl = loader.setup(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = loader.setup(test_ds, batch_size=batch_size, shuffle=shuffle)

    return train_ds, test_ds, train_dl, test_dl


####################################################################
# Adapted from https://flax.readthedocs.io/en/latest/getting_started.html
####################################################################
@struct.dataclass
class Metrics(training_state.MetricsCollection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


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
    return main_experiment(default_config())


def main_experiment(config):
    seed = config["seed"]

    # Get the dataset
    dataset_config = config["dataset"]
    train_ds, test_ds, train_dl, test_dl = get_data(dataset_config, seed)

    # Construct the model
    model_config = config["model"]
    model_type = model_config["type"]
    hidden_layers = model_config["hidden_layers"]
    activations = model_config["activations"]
    model, params = construct.model(
        model_type, seed, train_ds, hidden_layers, activations,
        jax.nn.initializers.glorot_normal(),
    )

    # Construct the optimiser
    optim_config = model_config["optim"]
    opt_type = optim_config["type"]
    opt_args = optim_config["args"]
    opt_kwargs = optim_config["kwargs"]
    optim = construct.optim(opt_type, opt_args, opt_kwargs)

    # Construct the training state
    init_rng = jax.random.key(seed)
    state = training_state.create(
        model, init_rng, Metrics, train_ds[0][0], optim,
    )
    del init_rng

    epochs = config["epochs"]
    data = experiment_loop(epochs, state, train_dl, test_dl)

    return data


def experiment_loop(epochs, state, train_dl, test_dl):
    metrics_history = {}
    for key in state.metrics.keys():
        metrics_history[f"train_{key}"] = []
        metrics_history[f"test_{key}"] = []

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
        print(f"==> Epoch {epoch}:")
        for type_ in ("train", "test"):
            for metric in state.metrics.keys():
                key = f"{type_}_{metric}"
                value = metrics_history[key][-1]
                print(f"\t{type_.title()} {metric.title()}:\t {value:.3f}")

    return {}

# config_file = "config/dense.yaml"
# with open(config_file, "r") as infile:
#     config = yaml.safe_load(infile)
# main_experiment(config)

# working_experiment()
