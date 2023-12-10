"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License. 
"""

from functools import partial
from pprint import pprint
from copy import deepcopy
import flax.linen as nn
import jax
import sparseeg.approximator as approximator
import jaxpruner
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
def optim(type_: str, config: dict, batch_size: int, ds_size: int):
    type_ = type_.lower()

    if type_ in ("adam", "rmsprop", "adagrad", "sgd"):
        return None, _optim_optax(type_, config)
    elif type_ in ("set", "magnitudepruning"):
        sparsity_updater = _optim_jaxpruner(type_, config, batch_size, ds_size)

        wrapped_config = config["wrapped"]
        wrapped_type = wrapped_config["type"].lower()
        wrapped = _optim_optax(wrapped_type, wrapped_config)

        return sparsity_updater, sparsity_updater.wrap_optax(wrapped)
    else:
        raise NotImplementedError(f"optim {type_} not implemented")


def _optim_jaxpruner(type_: str, config: dict, batch_size, ds_size):
    config = deepcopy(config)  # To avoid silent mutations
    args = config["args"]
    kwargs = config["kwargs"]

    if type_ == "set":
        # Create the sparsity distribution
        dist_config = kwargs["sparsity_distribution_fn"]
        kwargs["sparsity_distribution_fn"] = _jaxpruner_sparsity_dist(
            dist_config,
        )

        # Create the schedule at which we induce sparsity
        schedule_config = kwargs["scheduler"]
        kwargs["scheduler"] = _jaxpruner_scheduler(
            schedule_config, batch_size, ds_size
        )

        if "drop_fraction_fn" in kwargs.keys():
            kwargs["drop_fraction_fn"] = eval(kwargs["drop_fraction_fn"])

        return jaxpruner.SET(*args, **kwargs)

    elif type_ == "magnitudepruning":
        # Create the sparsity distribution
        dist_config = kwargs["sparsity_distribution_fn"]
        kwargs["sparsity_distribution_fn"] = _jaxpruner_sparsity_dist(
            dist_config,
        )

        # Create the schedule at which we induce sparsity
        schedule_config = kwargs["scheduler"]
        kwargs["scheduler"] = _jaxpruner_scheduler(
            schedule_config, batch_size, ds_size
        )

        return jaxpruner.MagnitudePruning(*args, **kwargs)

    elif type_ == "randompruning":
        # Create the sparsity distribution
        dist_config = kwargs["sparsity_distribution_fn"]
        kwargs["sparsity_distribution_fn"] = _jaxpruner_sparsity_dist(
            dist_config,
        )

        # Create the schedule at which we induce sparsity
        schedule_config = kwargs["scheduler"]
        kwargs["scheduler"] = _jaxpruner_scheduler(schedule_config)

        return jaxpruner.RandomPruning(*args, **kwargs)
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


def _jaxpruner_scheduler(config, batch_size, ds_size):
    type_ = config["type"].lower()
    args = config["args"]
    kwargs = config["kwargs"]
    if type_ == "periodicschedule":
        if batch_size > 0:
            # Adjust for batch size/epochs
            epochs = kwargs["update_freq"]
            kwargs["update_freq"] *= int(ds_size / batch_size)
        print(
            f"Sparsity updating every {epochs} epochs " +
            f"= {kwargs['update_freq']} gradient steps",
        )
        return jaxpruner.PeriodicSchedule(*args, **kwargs)
    elif type_ == "oneshotschedule":
        if batch_size > 0:
            # Adjust for batch size/epochs
            epochs = args[0]
            args[0] *= int(ds_size / batch_size)
        print(
            f"Sparsity updating every {epochs} epochs = {args[0]} " +
            "gradient steps",
        )
        return jaxpruner.OneShotSchedule(*args, **kwargs)
    elif type_ == "noupdateschedule":
        return jaxpruner.NoUpdateShotSchedule(*args, **kwargs)
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
