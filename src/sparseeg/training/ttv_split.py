import numpy as np
from copy import deepcopy
import hashlib
from clu import metrics
import jax
import os
import pickle

import sparseeg.data.dataset as dataset
import sparseeg.data.loader as loader
import sparseeg.training.state as training_state
import torch
from math import gcd
import warnings
from pprint import pprint


class TTVSplitTrainer:
    """
    Training on Train-Test-Validation Split
    """
    def __init__(
        self, experiment_loop, config, model_fn,
        optim_fn, dataset_fn, batch_size, shuffle, splitter, train_percent,
        valid_percent,
    ):
        self.experiment_loop = experiment_loop

        self._model_config = config["model"]
        self._model_fn = model_fn
        self._optim_config = config["model"]["optim"]
        self._optim_fn = optim_fn
        self._dataset_config = config["dataset"]
        self._dataset_fn = dataset_fn

        self._batch_size = batch_size

        self._shuffle = shuffle
        self._splitter = splitter

        if isinstance(train_percent, int):
            train_percent /= 100
        self._train_percent = train_percent

        if isinstance(valid_percent, int):
            valid_percent /= 100
        self._valid_percent = valid_percent

        self._save_data = {}

    def _split(self, ds, seed):
        splitter = self._splitter(
            self._train_percent, self._valid_percent, self._shuffle, seed,
        )

        train_ds, test_ds, valid_ds = splitter.split(ds)

        batch_size, train_ds = adjust_for_batch_size(
            self._batch_size, train_ds,
        )

        train_dl = loader.setup(
            train_ds, batch_size=batch_size, shuffle=self._shuffle
        )

        return train_ds, train_dl, test_ds, valid_ds

    def run(self, seed, epochs, weighted_loss, verbose=False):
        key = f"seed_{seed}"

        ds = self._dataset_fn(self._dataset_config, seed)
        splitted = self._split(ds, seed)
        train_ds, train_dl, test_ds, valid_ds = splitted

        model = self._model_fn(
            self._model_config, seed, train_ds,
        )
        optim = self._optim_fn(self._optim_config)

        data = self.experiment_loop(
            self, seed, epochs, model, optim, train_ds, train_dl,
            test_ds, valid_ds, weighted_loss, verbose=verbose,
        )

        self._save_data[key] = data

        print("TTV loop completed")
        return self._save_data


# Adjusts dataset size to account for batch size
def adjust_for_batch_size(batch_size, ds):
    if batch_size == 0:
        return len(ds), ds
    if len(ds) % batch_size != 0:
        ds = deepcopy(ds)
        x, y = ds.data
        ds_len = len(ds)
        x = x[:batch_size * (len(ds) // batch_size)]
        y = y[:batch_size * (len(ds) // batch_size)]

        ds.data = (x, y)

        print("\033[33m")
        warnings.warn(
            f"number of samples {ds_len} is not divisible by " +
            f"batch size {batch_size}, trimming extra dataset samples: " +
            f"new dataset size is {len(ds)}"
        )
        print("\033[0m")
    return batch_size, ds
