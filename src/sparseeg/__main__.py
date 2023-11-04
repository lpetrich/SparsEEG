#!/usr/bin/env python3
"""
Entry module for this package

You can run this package with
    python -m sparseeg -s "SAVE_DIR" -i "INT" default.py dense.yaml
"""

from importlib import import_module

import sparseeg

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
def run(experiment_file, config_file, index, save_at):
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
        os.makedirs(save_at)

    # Run the experiment
    data = experiment_module.main_experiment(config, save_file)
    data = {hyper.index_of(full_config, config): data}

    # # Save output data with orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(data)
    orbax_checkpointer.save(
        save_file, data, save_args=save_args
    )


if __name__ == "__main__":
    try:
        result = run()
    except Exception as e:
        print(f"Whoops! An error occured: {e}, {type(e)}")
    else:
        print(f"Result: {result}")
