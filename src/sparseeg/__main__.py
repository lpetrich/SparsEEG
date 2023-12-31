#!/usr/bin/env python3
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

from importlib import import_module

import sparseeg

from time import time
import orbax
import flax
from flax.training import orbax_utils
import click
import os
import pickle
import sparseeg.util.hyper as hyper
import yaml


# Use orbax for model saving
flax.config.update('flax_use_orbax_checkpointing', True)


@click.command()
@click.argument("experiment_file")
@click.argument("config_file")
@click.option(
    "-i", "--index", type=int, help="index of the hyper setting to run",
)
@click.option(
    "-s", "--save_at", type=click.Path(), help="path to save data at",
)
@click.option(
    "-c", "--cache-datasets", is_flag=True, default=False,
    help="cache the train, validation, and testing datasets for further " +
    "analysis offline",
)
def run(experiment_file, config_file, index, save_at, cache_datasets):
    # First check that config and experiment files exist
    fpath = "./src/sparseeg"
    config_path = f"{fpath}/config/{config_file}"
    if not os.path.exists(config_path):
        raise Exception(f"Config file does not exist: {config_file}")

    experiment_path = f"{fpath}/experiment/{experiment_file}"
    if not os.path.exists(experiment_path):
        raise Exception(f"Experiment file does not exist: {experiment_file}")

    # Parse config file
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)
        full_config = config
    config, _ = hyper.sweeps(config, index)

    # Import the experiment module
    # experiment_module_name = experiment_file.replace("/", ".")
    experiment_module_name = experiment_file.removesuffix(".py")
    experiment_module_name = f"sparseeg.experiment.{experiment_module_name}"
    globals()["experiment_module"] = import_module(experiment_module_name)

    # Create save directory
    save_at = os.path.join(save_at, config["save_dir"])
    save_file = os.path.join(save_at, f"{index}.pkl")
    if not os.path.isdir(save_at):
        try:
            os.makedirs(save_at)
        except FileExistsError:
            pass
    if os.path.exists(save_file):
        new_file = save_file + f".old_{int(time())}"
        os.rename(save_file, new_file)

    # Run the experiment
    data = experiment_module.main_experiment(config, save_file, cache_datasets)
    data = {hyper.index_of(full_config, config): data}

    # # Save output data with orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(data)
    print(f"Saving at {save_file}")
    orbax_checkpointer.save(
        save_file, data, save_args=save_args
    )
    print(f"Saved")


if __name__ == "__main__":
    try:
        result = run()
    except Exception as e:
        print(f"Whoops! An error occured: {e}, {type(e)}")
    else:
        print(f"Result: {result}")
