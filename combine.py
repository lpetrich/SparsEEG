#!/usr/bin/env python

import click
import pickle
import os
import yaml
import sparseeg.util.hyper as hyper
import orbax
from flax.training import orbax_utils


@click.argument("config")
@click.argument("dir")
@click.option("-s", "--filename", default="combined")
@click.option("-f", "--force", is_flag=True)
@click.command()
def combine(dir, config, filename, force):
    fpath = "./src/sparseeg/config"
    config_path = os.path.join(fpath, config)
    if not os.path.exists(config_path):
        raise Exception(f"Config file does not exist: {config_path}")

    # Parse config file
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)

    data_dir = os.path.join(dir, config["save_dir"])
    files = os.listdir(data_dir)
    files = list(map(lambda f: os.path.join(data_dir, f), files))

    all_data = {}
    chptr = orbax.checkpoint.PyTreeCheckpointer()
    for f in files:
        d = chptr.restore(f)
        all_data = _combine(all_data, d, config)

    filename = os.path.join(data_dir, filename)
    if os.path.exists(filename) and not force:
        raise ValueError(f"{filename} exists, use -f/--force to overwrite")

    save_args = orbax_utils.save_args_from_target(all_data)
    chptr.save(filename, all_data, save_args=save_args)


def get_key(d):
    assert len(d.keys()) == 1
    return list(d.keys())[0]


def get_keys(d):
    return list(d.keys())


def _combine(all_data, d, config):
    key = get_key(d)
    hyper_index = hyper.index_of(config, d[key]["config"])

    if hyper_index not in all_data.keys():
        all_data[hyper_index] = d[key]
    else:
        seed_key = get_key(d[key]["data"])
        seed_keys = get_keys(all_data[hyper_index]["data"])

        if seed_key in seed_keys:
            raise ValueError(f"{seed_key} key already in data dict")

        all_data[hyper_index]["data"][seed_key] = d[key]["data"][seed_key]

    return all_data


if __name__ == "__main__":
    combine()
