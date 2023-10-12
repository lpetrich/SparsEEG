#!/usr/bin/env python3
"""
Entry module for this package

You can run this package with
    python3 -m project
"""

from importlib import import_module
import click
import os
import pickle
import util.hyper as hyper
import yaml


@click.command()
@click.argument("experiment_file", type=click.Path(exists=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "-i", "--index", type=int, help="index of the hyper setting to run",
)
@click.option(
    "-s", "--save_at", type=click.Path(), help="path to save data at",
)
def run(experiment_file, config_file, index, save_at):
    # Parse config file
    with open(config_file, "r") as infile:
        config = yaml.safe_load(infile)
    config = hyper.sweeps(config, index)

    # Import the experiment module
    experiment_module_name = experiment_file.replace("/", ".")
    experiment_module_name = experiment_module_name.removesuffix(".py")
    globals()["experiment_module"] = import_module(experiment_module_name)

    # Run the experiment
    data = experiment_module.main_experiment(config)

    # Save output data here
    if not os.path.isdir(save_at):
        os.makedirs(save_at)
    save_file = os.path.join(save_at, f"{index}.pkl")
    with open(save_file, "wb") as outfile:
        pickle.dump(data, outfile)


if __name__ == "__main__":
    run()
