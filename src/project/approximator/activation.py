from jax import jit

# Identity activation
@jit
def identity(x):
    return x
