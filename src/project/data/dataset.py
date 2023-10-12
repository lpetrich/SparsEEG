import jax
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset

class ClassifierDataset(Dataset):
    def __init__(self, rng, n_samples, x_dim, n_classes, W, b):
        if W is None:
            rng, w_key = jax.random.split(rng)
            W = jax.random.normal(w_key, (x_dim, n_classes))
        if b is None:
            rng, b_key = jax.random.split(rng)
            b = jax.random.normal(b_key, (n_classes,))

        self.n_samples = n_samples
        self._W = W
        self._b = b
        self._n_classes = n_classes

        key, key_sample, key_noise = jax.random.split(rng, 3)
        x_samples = jax.random.normal(key_sample, (self.n_samples, x_dim))

        # Y ~ Multinoulli(π = softmax(Wᵀx))
        y_samples = jax.random.categorical(
            key_noise, jnp.dot(x_samples, W), axis=-1,
        )

        self.x_samples = np.array(x_samples)
        self.y_samples = np.array(y_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            np.array(self.x_samples[index]),
            np.array(self.y_samples[index]),
        )

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b


def load(identifier: str, seed):
    train_init_rng = jax.random.key(seed)
    test_init_rng = jax.random.key(seed)

    if identifier.lower() == "classifierdataset-default":
        x_dim = 10
        n_classes = 5
        train_ds = ClassifierDataset(
            train_init_rng, 100, x_dim, n_classes, None, None,
        )
        test_ds = ClassifierDataset(
            test_init_rng, 20, x_dim, n_classes, None, None,
        )
    else:
        raise NotImplementedError(f"{identifier} does not exist")

    del train_init_rng
    del test_init_rng

    return train_ds, test_ds
