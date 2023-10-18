import numpy as np
from clu import metrics
import jax
# import src.project.data.dataset as dataset
# import src.project.data.loader as loader
# import src.project.training.state as training_state
import data.dataset as dataset
import data.loader as loader
import training.state as training_state
import torch
from math import gcd
import warnings
from pprint import pprint


class NestedCrossValidation:
    """
    Performs nested cross validation using some dataset by splitting that
    dataset repeatedly and calling an experiment loop on the split data.

    This class will perform nested cross validation on a dataset. The dataset
    is loaded using a function `dataset_fn(seed)`. The dataset is split for the
    nested cross validation scheme using the `splitter` parameter, of
    type`DatasetSplitter`. This parameter splits the dataset at both the
    internal and external CV loops.

    The experiment to run on each of the inner CV loops is determined by the
    `experiment_loop` parameter. This parameters is a function which takes the
    following arguments:

        1. Number of epochs over the dataset to train at each inner CV loop
        2. The training state (see src.project.training.state)
        3. A `Dataloader` which loads the training data
        3. A `Dataloader` which loads the testing/validation data
        4. A verbose flag, indicating if debugging info should be printed

    This experiment loop should take care of training using the training
    dataloader and validating/testing using the testing dataloader, and
    recording any information to save during the epoch. The data returned by
    the experiment loop is cached during each loop of the CV procedure.
    """
    def __init__(
        self, experiment_loop, model_fn, optim_fn, n_external_folds,
        n_internal_folds, dataset_fn, external_batch_size, internal_batch_size,
        shuffle_external, shuffle_internal, splitter
    ):
        self.experiment_loop = experiment_loop

        self._model_fn = model_fn
        self._optim_fn = optim_fn
        self._dataset_fn = dataset_fn

        self._n_external_folds = n_external_folds
        self._n_internal_folds = n_internal_folds

        self._external_batch_size = external_batch_size
        self._internal_batch_size = internal_batch_size

        self._shuffle_external = shuffle_external
        self._shuffle_internal = shuffle_internal

        self._splitter = splitter

        self._save_data = {}

    def _split_external(self, ds, fold):
        splittable_ds = self._splitter(ds, self._n_external_folds)
        train_ds, test_ds = splittable_ds.split(fold)

        batch_size = adjust_batch_size(self._external_batch_size, train_ds)

        # Set up the dataset loader
        train_dl = loader.setup(
            train_ds, batch_size=batch_size, shuffle=self._shuffle_external,
        )

        # Test batch size is always the entire test set
        test_batch_size = len(test_ds)
        test_dl = loader.setup(
            test_ds, batch_size=test_batch_size,
            shuffle=self._shuffle_external,
        )

        return train_ds, test_ds, train_dl, test_dl

    def _split_internal(self, ext_train_ds, fold):
        splittable_ext_ds = self._splitter(
            ext_train_ds, self._n_internal_folds,
        )
        train_ds, test_ds = splittable_ext_ds.split(fold)

        batch_size = adjust_batch_size(
            self._internal_batch_size, train_ds,
        )

        train_dl = loader.setup(
            train_ds, batch_size=batch_size,
            shuffle=self._shuffle_internal,
        )
        test_dl = loader.setup(
            test_ds, batch_size=len(test_ds),  # always full batch
            shuffle=self._shuffle_internal
        )

        return train_ds, train_dl, test_ds, test_dl

    def run(self, seed, epochs, verbose=False):
        """
        Run the nested cross validation procedure
        """
        assert f"seed_{seed}" not in self._save_data.keys()
        key = f"seed_{seed}"
        self._save_data[key] = {}

        save_data = self._save_data[key]
        for i in range(self._n_external_folds):

            # Get the dataset
            ext_ds = self._dataset_fn(seed)

            # Split the dataset into folds
            ext_ds = self._split_external(ext_ds, i)
            ext_train_ds, ext_test_ds, ext_train_dl, ext_test_dl = ext_ds

            # Internal CV loop
            save_data[f"external_fold_{i}"] = {}
            for j in range(self._n_internal_folds):
                # Split the fold into multiple sub-folds
                train_ds, train_dl, test_ds, test_dl = self._split_internal(
                    ext_train_ds, j
                )

                # Construct a new model for each fold
                model = self._model_fn(seed, train_ds)

                # Construct a new optimiser for each fold
                optim = self._optim_fn()

                data = self.experiment_loop(
                    seed, epochs, model, optim, train_ds, train_dl, test_ds,
                    test_dl, verbose=verbose,
                )

                save_data[f"external_fold_{i}"][f"internal_fold_{j}"] = data

        # Train/Test on all data
        model = self._model_fn(seed, train_ds)
        optim = self._optim_fn()
        data = self.experiment_loop(
            seed, epochs, model, optim, ext_train_ds, ext_train_dl,
            ext_test_ds, ext_test_dl, verbose=verbose,
        )
        save_data["external"] = data

        # pprint(save_data["external"]["test_accuracy"])
        return self._save_data


def adjust_batch_size(batch_size, ds):
    if batch_size == 0:
        return len(ds)
    if len(ds) % batch_size != 0:
        old_batch_size = batch_size
        batch_size = gcd(batch_size, len(ds))
        print("\033[33m")
        warnings.warn(
            f"number of samples {len(ds)} is not divisible by " +
            f"batch size {old_batch_size}, using batch size {batch_size} " +
            "instead",
        )
        print("\033[0m")
    return batch_size
