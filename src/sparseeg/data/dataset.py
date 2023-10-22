import jax
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
import sklearn.model_selection as model_selection
from sklearn.datasets import load_wine
from copy import deepcopy
from abc import ABC, abstractmethod


class DatasetSplitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def split(self, fold):
        pass


class StratifiedKFold(DatasetSplitter):
    def __init__(self, dataset, n_folds):
        self.n_folds = n_folds
        self.dataset = dataset

    def split(self, fold):
        skf = model_selection.StratifiedKFold(n_splits=self.n_folds)
        return self._split(skf, fold)

    def _split(self, kf, fold):
        train = deepcopy(self.dataset)
        test = deepcopy(self.dataset)

        # Create the stratified splits
        splits = kf.split(self.dataset.data[0], self.dataset.data[1])
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
        train.data = (train_x_samples, train_y_samples)

        # Create the new testing set
        test_x_samples = test.x_samples[test_indices, :][0]
        test_y_samples = test.y_samples[test_indices][0]
        test.data = (test_x_samples, test_y_samples)

        return train, test


class WAYEEGGALDataset(Dataset):
    def __init__(self):
        print(f"Loading WAL-EEG-GAL Dataset of type: {Dataset}")
        data = load_wine()
        self.x_samples = data["data"]
        self.y_samples = data["target"]
        self._n_classes = 3

    def __len__(self):
        return len(self.y_samples)

    def __getitem__(self, index):
        return (
            self.x_samples[index, :],
            self.y_samples[index],
        )

    @property
    def n_samples(self):
        return len(self)

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


class WineDataset(Dataset):
    def __init__(self):
        data = load_wine()
        self.x_samples = data["data"]
        self.y_samples = data["target"]
        self._n_classes = 3

    def __len__(self):
        return len(self.y_samples)

    def __getitem__(self, index):
        return (
            self.x_samples[index, :],
            self.y_samples[index],
        )

    @property
    def n_samples(self):
        return len(self)

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


class ClassifierDataset(Dataset):
    def __init__(self, rng, n_samples, x_dim, n_classes):
        self._n_classes = n_classes

        key, key_sample, key_noise = jax.random.split(rng, 3)
        x_samples = jax.random.normal(key_sample, (n_samples, x_dim))
        x_samples = np.array(x_samples)

        # Two populations:
        #   𝒩([1, 1], [1, 1])           Label: 0
        #   𝒩([0, 0], [0.75, 0.25])     Label: 1
        x_samples[:len(x_samples) // 2, :] += 1
        x_samples[len(x_samples) // 2:, 0] *= 0.75
        x_samples[len(x_samples) // 2:, 1] *= 0.25

        # Y ~ Multinoulli(π = softmax(Wᵀx))
        y_samples = np.zeros(x_samples.shape[0], dtype=np.int)
        y_samples[:len(y_samples) // 2] = 0
        y_samples[len(y_samples) // 2:] = 1

        self.x_samples = x_samples
        self.y_samples = np.array(y_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            np.array(self.x_samples[index]),
            np.array(self.y_samples[index]),
        )

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
    identifier = identifier.lower()

    if identifier == "classifierdataset-default":
        x_dim = 10
        n_classes = 3
        train_ds = ClassifierDataset(train_init_rng, 200, x_dim, n_classes)
    elif identifier == "winedataset":
        train_ds = WineDataset()
    elif identifier == "wayeeggaldataset":
        print("Initializing WAY-EEG-GAL Dataset... JK")
        train_ds = WAYEEGGALDataset()
    else:
        raise NotImplementedError(f"{identifier} does not exist")

    del train_init_rng

    return train_ds