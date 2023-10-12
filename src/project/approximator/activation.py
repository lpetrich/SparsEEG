from jax import jit


@jit
def identity(x):
    return x
