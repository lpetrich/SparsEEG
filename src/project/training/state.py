from flax import struct
from jax import jit, grad
from flax.training import train_state
import optax
from typing import Sequence, Callable
from abc import ABC, abstractmethod
from clu import metrics
from typing import Any
import jaxpruner


class MetricsCollection(ABC, metrics.Collection):
    pass

    def keys(self):
        return [metric for metric, _ in self.compute().items()]


####################################################################
# Adapted from https://flax.readthedocs.io/en/latest/getting_started.html
####################################################################
@struct.dataclass
class TrainState(train_state.TrainState):
    """
    Initializes model parameters and tracks optimizer state, model
    state/parameters, and metrics to store during training
    """
    metrics: MetricsCollection
    sparsity_updater: jaxpruner.BaseUpdater = struct.field(pytree_node=False)

    def update_sparsity(self):
        if self.sparsity_updater is not None:
            return self.sparsity_updater.post_gradient_update(
                self.params, self.opt_state,
            )
        else:
            return self.params


def create(module, rng, metrics_type, dummy_input, opt):
    params = module.init(rng, dummy_input)["params"]
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=opt[1],
        metrics=metrics_type.empty(), sparsity_updater=opt[0],
    )


@jit
def step(state, batch):
    # TODO: Eventually, we want to factor loss_fn out
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
