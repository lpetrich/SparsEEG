from jax import jit, grad
from flax.training import train_state
import optax
from typing import Sequence, Callable
from abc import ABC, abstractmethod
from clu import metrics

class MetricsCollection(ABC,metrics.Collection):
    pass

    def keys(self):
        return [metric for metric, _ in self.compute().items()]

####################################################################
# Adapted from https://flax.readthedocs.io/en/latest/getting_started.html
####################################################################
class TrainState(train_state.TrainState):
    metrics: MetricsCollection


def create(module, rng, metrics_type, dummy_input, opt):
    params = module.init(rng, dummy_input)["params"]
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=opt,
        metrics=metrics_type.empty(),
    )


@jit
def step(state, batch):
    # Eventually, we want to factor loss_fn out
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['inputs'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['labels']
        ).mean()
        return loss
    grad_fn = grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state
####################################################################
