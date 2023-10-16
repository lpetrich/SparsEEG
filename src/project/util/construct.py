from functools import partial
from pprint import pprint
from copy import deepcopy
import flax.linen as nn
import jax
import jaxpruner
import src.project.approximator as approximator
import optax
import warnings


def model(type_, seed, ds, *args, **kwargs):
    init_key = jax.random.key(seed)
    if type_.lower() == "densemlp":
        return _construct_dense_mlp(init_key, ds, *args, **kwargs)
    else:
        raise NotImplementedError(f"model {type_} not implemented")


def _get_activations(acts, kwargs=None):
    if kwargs is not None:
        return [_get_activation(acts[i], kwargs[i]) for i in range(len(acts))]
    else:
        return [_get_activation(act) for act in acts]


def _get_activation(act: str, kwargs={}):
    if act == "relu":
        return nn.relu
    elif act == "tanh":
        return nn.tanh
    elif act == "hard_tanh":
        return nn.hard_tanh
    elif act == "hard_sigmoid":
        return nn.hard_sigmoid
    elif act == "sigmoid":
        return nn.sigmoid
    elif act == "softmax":
        return lambda x: nn.softmax(x, **kwargs)
    elif act == "identity":
        return lambda x: x
    else:
        raise NotImplementedError(f"unknown activation {act}")


def _construct_dense_mlp(
    init_key, ds, hidden_sizes, activations, weight_init
):
    # Deepcopy parameters to avoid silent mutations later
    activations = deepcopy(activations)
    hidden_sizes = deepcopy(hidden_sizes)

    assert len(hidden_sizes) == len(activations)
    x = ds[0][0]

    # Append a final output layer to the network architecture
    classes = ds.n_classes
    layer_sizes = [*hidden_sizes, ds.n_classes]
    activations.append("identity")
    activations = _get_activations(activations)

    model = approximator.MLP(
        features=layer_sizes,
        act=activations,
        weight_init=weight_init,
    )

    return model


# Returns (sparsity wrapper, optimiser)
def optim(type_: str, config: dict):
    type_ = type_.lower()

    if type_ in ("adam", "rmsprop", "adagrad", "sgd"):
        return None, _optim_optax(type_, config)
    elif type_ in ("set"):
        sparsity_updater = _optim_jaxpruner(type_, config)
        # TODO: assign sparsity scheduler... right now it doesn't run I don't
        # think

        wrapped_config = config["wrapped"]
        wrapped_type = wrapped_config["type"].lower()
        wrapped = _optim_optax(wrapped_type, wrapped_config)

        return sparsity_updater, sparsity_updater.wrap_optax(wrapped)
    else:
        raise NotImplementedError(f"optim {type_} not implemented")


def _optim_jaxpruner(type_: str, config: dict):
    config = deepcopy(config)  # To avoid silent mutations
    if type_ == "set":
        args = config["args"]
        kwargs = config["kwargs"]

        # SET prunes after each epoch
        dist_config = kwargs["sparsity_distribution_fn"]
        kwargs["sparsity_distribution_fn"] = _jaxpruner_sparsity_dist(
            dist_config,
        )
        schedule_config = kwargs["scheduler"]
        kwargs["scheduler"] = _jaxpruner_scheduler(schedule_config)

        if "drop_fraction_fn" in kwargs.keys():
            kwargs["drop_fraction_fn"] = eval(kwargs["drop_fraction_fn"])

        return jaxpruner.SET(*args, **kwargs)
    else:
        raise NotImplementedError(f"sparse optim {type_} not implemented")


def _jaxpruner_sparsity_dist(config):
    type_ = config["type"].lower()
    if "args" in config:
        warnings.warn("key 'args' ignored for sparsity distributions")
    kwargs = config["kwargs"]

    if type_ == "erk":
        return partial(jaxpruner.sparsity_distributions.erk, **kwargs)
    elif type_ == "uniform":
        return partial(jaxpruner.sparsity_distributions.uniform, **kwargs)
    else:
        raise NotImplementedError(f"sparisty dist {type_} not implemented")


def _jaxpruner_scheduler(config):
    type_ = config["type"].lower()
    args = config["args"]
    kwargs = config["kwargs"]
    if type_ == "periodicschedule":
        return jaxpruner.PeriodicSchedule(*args, **kwargs)
    elif type_ == "oneshotschedule":
        return jaxpruner.OneShotSchedule(*args, **kwargs)
    elif type_ == "noupdateschedule":
        return jaxpruner.NoUpdateShotSchedule(*args, **kwargs)
    elif type_ == "polynomialschedule":
        return jaxpruner.PolynomialSchedule(*args, **kwargs)
    else:
        raise NotImplementedError(f"schedule {type_} not implemented")


# For parameters that each type of optimiser expects, see
# https://optax.readthedocs.io/en/latest/api.html
def _optim_optax(type_: str, config: dict):
    args = config["args"]
    kwargs = config["kwargs"]

    if type_.lower() == "adam":
        return optax.adam(*args, **kwargs)
    if type_.lower() == "rmsprop":
        return optax.rmsprop(*args, **kwargs)
    if type_.lower() == "adagrad":
        return optax.adagrad(*args, **kwargs)
    if type_.lower() == "sgd":
        return optax.sgd(*args, **kwargs)
    else:
        raise NotImplementedError(f"optim {type_} not implemented")
