import jax
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy


class ClassifierDataset(Dataset):
    def __init__(self, rng, n_samples, x_dim, n_classes, W, b):
        if W is None:
            rng, w_key = jax.random.split(rng)
            W = jax.random.normal(w_key, (x_dim, n_classes))
        if b is None:
            rng, b_key = jax.random.split(rng)
            b = jax.random.normal(b_key, (n_classes,))

        self._W = W
        self._b = b
        self._n_classes = n_classes

        key, key_sample, key_noise = jax.random.split(rng, 3)
        x_samples = jax.random.normal(key_sample, (n_samples, x_dim))

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

    def stratify_kfold(self, n_folds, fold):
        train = deepcopy(self)
        test = deepcopy(self)

        # Create the stratified splits
        skf = StratifiedKFold(n_splits=n_folds)
        splits = skf.split(self.data[0], self.data[1])
        train_indices = []
        test_indices = []
        for fold_num, (train_ind, test_ind) in enumerate(splits):
            if fold_num != fold:
                # This is really ugly...
                continue
            train_indices.append(train_ind)
            test_indices.append(test_ind)

        # Create the new training set
        train_x_samples = train.x_samples[np.array(train_indices), :][0]
        train_y_samples = train.y_samples[train_indices][0]
        print(train_x_samples.shape)
        print(train.x_samples.shape)
        train.data = (train_x_samples, train_y_samples)

        # Create the new testing set
        test_x_samples = test.x_samples[test_indices, :][0]
        test_y_samples = test.y_samples[test_indices][0]
        test.data = (test_x_samples, test_y_samples)

        return train, test

    @property
    def n_samples(self):
        return len(self.x_samples)

    @property
    def data(self):
        return (self.x_samples, self.y_samples)

    @data.setter
    def data(self, data):
        self.x_samples = data[0]
        self.y_samples = data[1]

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b


# Load a full dataset with pre-defined train/test splits
def load_train_test(identifier: str, seed):
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


# Load a full dataset
def load(identifier: str, seed):
    train_init_rng = jax.random.key(seed)

    if identifier.lower() == "classifierdataset-default":
        x_dim = 10
        n_classes = 5
        train_ds = ClassifierDataset(
            train_init_rng, 120, x_dim, n_classes, None, None,
        )
    else:
        raise NotImplementedError(f"{identifier} does not exist")

    del train_init_rng

    return train_ds
