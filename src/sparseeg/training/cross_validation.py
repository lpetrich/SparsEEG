import numpy as np
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
        self, experiment_loop, checkpoint_dir, save_file, config, model_fn,
        optim_fn, n_external_folds, n_internal_folds, dataset_fn,
        external_batch_size, internal_batch_size, shuffle_external,
        shuffle_internal, splitter
    ):
        self.experiment_loop = experiment_loop

        # Checkpointing stuff: we save the checkpoint data at the hash digest
        # of the configuration file
        self._checkpoint_dir = checkpoint_dir
        self._data_save_file = save_file  # Where data will be stored
        self._config = config
        m = hashlib.sha256()
        m.update(str(self._config).encode())
        self._digest = str(m.hexdigest())
        self._checkpoint_filename = os.path.join(
            self._checkpoint_dir, f"{self._digest}.pkl",
        )
        self._current_external_fold = 0
        self._current_internal_fold = 0
        self._completed = False

        self._model_config = config["model"]
        self._model_fn = model_fn
        self._optim_config = config["model"]["optim"]
        self._optim_fn = optim_fn
        self._dataset_config = config["dataset"]
        self._dataset_fn = dataset_fn

        self._n_external_folds = n_external_folds
        self._n_internal_folds = n_internal_folds

        self._external_batch_size = external_batch_size
        self._internal_batch_size = internal_batch_size

        self._shuffle_external = shuffle_external
        self._shuffle_internal = shuffle_internal

        self._splitter = splitter

        self._save_data = {}

    def load_checkpoint(self):
        if os.path.isfile(self._checkpoint_filename):
            print("loading checkpoint file:", self._checkpoint_filename)
            with open(self._checkpoint_filename, "rb") as infile:
                cv = pickle.load(infile)

            print("Loaded checkpoint:")
            print(f"\tCV loop completed:\t {cv._completed}")
            print(
                f"\tExternal Fold:\t {cv._current_external_fold}/" +
                f"{cv._n_external_folds}",
            )
            print(
                f"\tInternal Fold:\t {cv._current_internal_fold}/" +
                f"{cv._n_internal_folds}",
            )

            return cv

        return self

    def checkpoint(self):
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
        with open(self._checkpoint_filename, "wb") as outfile:
            pickle.dump(self, outfile)

    def _split_external(self, ds, fold):
        splittable_ds = self._splitter(ds, self._n_external_folds)
        train_ds, test_ds = splittable_ds.split(fold)

        # TODO: instead of adjuting batch size, should we instead just skip the
        # remainder data?
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

        # TODO: instead of adjuting batch size, should we instead just skip the
        # remainder data?
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

    def run(self, seed, epochs, save_file, verbose=False):
        """
        Run the nested cross validation procedure
        """
        print("Running Nested CV")

        if self._completed:
            print("CV loop completed, returning cached data")
            return self._save_data

        if f"seed_{seed}" in self._save_data.keys():
            warnings.warn(
                f"key seed_{seed} already found in dict, continuing " +
                f"adding to seed data",
            )

        key = f"seed_{seed}"
        self._save_data[key] = {}

        save_data = self._save_data[key]
        save_data["external"] = {}
        external_folds_to_run = range(
            self._current_external_fold, self._n_external_folds,
        )
        for i in external_folds_to_run:
            print(f"external fold {i} starting")
            # Get the dataset
            ext_ds = self._dataset_fn(self._dataset_config, seed)
            # Split the dataset into folds
            _ext_ds = self._split_external(ext_ds, i)
            ext_train_ds, ext_test_ds, ext_train_dl, ext_test_dl = _ext_ds

            # Internal CV loop
            save_data[f"external_fold_{i}"] = {}
            internal_folds_to_run = range(
                self._current_internal_fold, self._n_internal_folds,
            )
            for j in internal_folds_to_run:
                # Split the fold into multiple sub-folds
                _int_ds = self._split_internal(
                    ext_train_ds, j
                )
                _train_ds, _train_dl, _test_ds, _test_dl = _int_ds
                internal_train_ds = _train_ds
                internal_test_ds = _test_ds
                internal_train_dl = _train_dl
                internal_test_dl = _test_dl

                # Construct a new model for each fold
                model = self._model_fn(
                    self._model_config, seed, internal_train_ds,
                )

                # Construct a new optimiser for each fold
                optim = self._optim_fn(self._optim_config)

                # No caching is done during the experiment loop, only after
                # I.e., we cache on a fold-by-fold basis, rather than on an
                # epoch-by-epoch basis
                data = self.experiment_loop(
                    self, seed, epochs, model, optim, internal_train_ds,
                    internal_train_dl, internal_test_ds, internal_test_dl,
                    verbose=verbose,
                )

                save_data[f"external_fold_{i}"][f"internal_fold_{j}"] = data
                self._current_internal_fold = j + 1  # Set next fold to run

            # Restart internal fold numbering for the next external fold
            self._current_internal_fold = 0

            # Train on all training data and evaluate on validation data for
            # fold i
            # Get the dataset
            ext_ds = self._dataset_fn(self._dataset_config, seed)
            # Split the dataset into folds
            _ext_ds = self._split_external(ext_ds, i)
            ext_train_ds, ext_test_ds, ext_train_dl, ext_test_dl = _ext_ds

            # Train/Test on all data
            model = self._model_fn(
                self._model_config, seed, ext_train_ds,
            )
            optim = self._optim_fn(self._optim_config)

            # No caching is done during the experiment loop, only after
            # I.e., we cache on a fold-by-fold basis, rather than on an
            # epoch-by-epoch basis
            data = self.experiment_loop(
                self, seed, epochs, model, optim, ext_train_ds, ext_train_dl,
                ext_test_ds, ext_test_dl, verbose=verbose,
            )
            save_data["external"][f"fold_{i}"] = data
            self._current_external_fold = i + 1  # Set next fold to run

            print(self._current_external_fold)

        # pprint(save_data["external"]["test_accuracy"])
        self._completed = True
        print("CV loop completed")
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
