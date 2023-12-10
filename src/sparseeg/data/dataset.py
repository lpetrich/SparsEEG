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
import jax
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
import sklearn.model_selection as ms
from sklearn.datasets import load_wine
from sparseeg.data.csv_loader import load_wayeeggal
from copy import deepcopy
from abc import ABC, abstractmethod


class DatasetSplitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def split(self, fold):
        pass


class StratifiedTTV(DatasetSplitter):
    def __init__(self, train_percent, validation_percent, shuffle, seed):
        self.seed = seed
        self.train_percent = train_percent
        self.validation_percent = validation_percent
        self.test_percent = 1 - self.train_percent - self.validation_percent
        assert np.isclose(
            self.train_percent +
            self.test_percent +
            self.validation_percent, 1
        )
        self.shuffle = shuffle

    def split(self, ds):
        train_ds = deepcopy(ds)
        test_ds = deepcopy(ds)
        validation_ds = deepcopy(ds)

        if self.test_percent > 0:
            x_train, x_test, y_train, y_test = ms.train_test_split(
                train_ds.x_samples, train_ds.y_samples,
                stratify=train_ds.y_samples, random_state=self.seed,
                shuffle=self.shuffle, test_size=self.test_percent,
            )
            test_ds.data = (x_test, y_test)
        else:
            x_train, y_train = train_ds.x_samples, train_ds.y_samples

        x_train, x_valid, y_train, y_valid = ms.train_test_split(
            x_train, y_train, stratify=y_train, random_state=self.seed,
            shuffle=self.shuffle, test_size=self.validation_percent,
        )

        train_ds.data = (x_train, y_train)
        validation_ds.data = (x_valid, y_valid)

        print("Dataset Statistics:")
        print(f"\tTraining on {len(train_ds)} samples")
        print(f"\tTesting on {len(test_ds)} samples")
        print(f"\tValidating on {len(validation_ds)} samples")

        return train_ds, test_ds, validation_ds


class StratifiedKFold(DatasetSplitter):
    def __init__(self, ds, n_folds):
        self.n_folds = n_folds
        self.ds = ds

    def split(self, fold):
        skf = ms.StratifiedKFold(n_splits=self.n_folds)
        return self._split(skf, fold)

    def _split(self, kf, fold):
        train = deepcopy(self.ds)
        test = deepcopy(self.ds)

        # Create the stratified splits
        splits = kf.split(self.ds.data[0], self.ds.data[1])
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
    def __init__(self, trim_level, config, seed):
        self._trim_level = trim_level

        empty = config.get("empty", False)

        # Set the percent of each subject's data to train on
        if "percent" not in config:
            self._percent = 1.0
        else:
            percent = config["percent"]
            if isinstance(percent, int):
                percent /= 100
            self._percent = percent

        print(f"Initializing WAL-EEG-GAL Dataset of type: {Dataset}")

        # Get the number of subjects whose data to include
        print("=====================")
        print(f"Using seed: {seed}")
        print("=====================")
        rng = np.random.default_rng(seed=seed)
        self._rng = rng

        if not empty:
            if "n_subjects" in config.keys():
                n = config["n_subjects"]
                assert n > 0
                subjects = self._rng.choice(range(1, 11), n, replace=False)
            elif "subjects" in config.keys():
                subjects = config["subjects"]
            elif "subject" in config.keys():
                subjects = [config["subject"]]
            else:
                raise ValuerError(
                    "expected 'subjects', 'subject', or 'n_subjects' " +
                    "keys in config"
                )

            print(f"Using data from subjects {subjects}")

            # Load first subject's data
            self.x_samples, self.y_samples = self._load_subject_data(subjects[0])

            # Load next subjects' data and randomly subsample as above
            for subject in subjects[1:]:
                X, y = self._load_subject_data(subject)
                self.x_samples = np.concatenate((self.x_samples, X))
                self.y_samples = np.concatenate((self.y_samples, y))

            # Renumber labels from 0
            classes = np.unique(self.y_samples)
            for i, c in enumerate(classes):
                self.y_samples[self.y_samples == c] = i

            self._classes = np.unique(self.y_samples)

            print(f"Number of x samples: {self.x_samples.shape}")
            print(f"Number of y samples: {self.y_samples.shape}")
            print(f"Number of classes: {self.n_classes}")
            print(f"Targets: {self.classes}")
            for c in self._classes:
                print(f"\tClass {c} samples:", sum(self.y_samples == c))

    def _load_subject_data(
        self, subject, trim_level=None, percent=None, rng=None,
    ):
        if rng is None:
            rng = self._rng
        if trim_level is None:
            trim_level = self._trim_level
        if percent is None:
            percent = self._percent

        # Load data
        X, y = load_wayeeggal(
            subject=subject, train=True, return_X_y=True,
        )

        # Restrict to only specific labels
        X, y = self._restrict(X, y, trim_level)

        # Randomly subsample the data
        return _random_subset(rng, X, y, percent)

    def _restrict(self, X, y, level):
        if level.lower() == "full":
            return X, y
        if level.lower() == "kaggle":
            inds = np.where(
                (y == 2) |
                (y == 3) |
                (y == 5) |
                (y == 7) |
                (y == 8) |
                (y == 10),
                True, False,
            )
        elif level.lower() == "low":
            inds = np.where(
                (y == 2) |
                (y == 7) |
                (y == 8) |
                (y == 10),
                True, False,
            )

        return X[inds], y[inds]

    def __len__(self):
        return len(self.y_samples)

    def __getitem__(self, index):
        return (
            self.x_samples[index, :],
            self.y_samples[index],
        )

    def get_dataset_for(self, label):
        new = deepcopy(self)
        new.data = (
            self.x_samples[self.y_samples == label],
            self.y_samples[self.y_samples == label],
        )

        return new

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
        self._classes = np.unique(self.y_samples)

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def classes(self):
        return self._classes


class WineDataset(Dataset):
    def __init__(self):
        data = load_wine()
        self.x_samples = data["data"]
        self.y_samples = data["target"]
        self._classes = np.unique(self.y_samples)

    def __len__(self):
        return len(self.y_samples)

    def __getitem__(self, index):
        return (
            self.x_samples[index, :],
            self.y_samples[index],
        )

    def get_dataset_for(self, label):
        new = deepcopy(self)
        new.data = (
            self.x_samples[self.y_samples == label],
            self.y_samples[self.y_samples == label],
        )

        return new

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def classes(self):
        return self._classes

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


class ClassifierDataset(Dataset):
    def __init__(self, rng, n_samples, x_dim, n_classes):
        self.n_classes = n_classes

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

    def get_dataset_for(self, label):
        new = deepcopy(self)
        new.data = (
            self.x_samples[self.y_samples == label],
            self.y_samples[self.y_samples == label],
        )

        return new

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
def load(identifier: str, config, seed):
    train_init_rng = jax.random.key(seed)
    identifier = identifier.lower()

    if identifier == "classifierdataset-default":
        x_dim = 10
        n_classes = 3
        train_ds = ClassifierDataset(train_init_rng, 200, x_dim, n_classes)
    elif identifier == "winedataset":
        train_ds = WineDataset()
    elif identifier == "wayeeggaldataset":
        train_ds = WAYEEGGALDataset("kaggle", config, seed)
    elif identifier == "wayeeggaldataset-low":
        train_ds = WAYEEGGALDataset("low", config, seed)
    else:
        raise NotImplementedError(f"{identifier} does not exist")

    del train_init_rng

    return train_ds


def _random_subset(rng, X, y, percent: float):
    if percent == 1.0:
        return X, y

    n = int(len(y) * percent)
    ind = rng.integers(0, len(y), n)

    return X[ind, :], y[ind]
