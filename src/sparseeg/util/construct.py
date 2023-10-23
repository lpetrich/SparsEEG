from copy import deepcopy
import flax.linen as nn
import jax
import sparseeg.approximator as approximator
import optax


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


# For parameters that each type of optimiser expects, see
# https://optax.readthedocs.io/en/latest/api.html
def optim(type_: str, args: tuple, kwargs: dict):
    if type_.lower() == "adam":
        return optax.adam(*args, **kwargs)
    if type_.lower() == "rmsprop":
        return optax.rmsprop(*args, **kwargs)
    if type_.lower() == "adagrad":
        return optax.adagrad(*args, **kwargs)
    if type_.lower() == "sgd":
        return optax.sgd(*args, **kwargs)
    else:
        raise NotImplementedError(f"{type_} optimiser unknown")
