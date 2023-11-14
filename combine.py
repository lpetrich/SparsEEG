#!/usr/bin/env python

from tqdm import tqdm
import click
import pickle
import os
import yaml
import sparseeg.util.hyper as hyper
import orbax
from flax.training import orbax_utils
import fnmatch


@click.argument("config")
@click.argument("dir")
@click.option("-s", "--filename", default="combined")
@click.option("-f", "--force", is_flag=True)
@click.option("-i", "--ignore")
@click.command()
def combine(dir, config, filename, force, ignore):
    ignore = ignore.split(",")
    fpath = "./src/sparseeg/config"
    config_path = os.path.join(fpath, config)
    if not os.path.exists(config_path):
        raise Exception(f"Config file does not exist: {config_path}")

    # Parse config file
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)

    data_dir = os.path.join(dir, config["save_dir"])
    print("Combining data in", data_dir)

    filename = os.path.join(data_dir, filename)
    if os.path.exists(filename) and not force:
        raise ValueError(f"{filename} exists, use -f/--force to overwrite")

    files = os.listdir(data_dir)
    print("Files:", files)
    files = list(map(lambda f: os.path.join(data_dir, f), files))
    files = fnmatch.filter(files, "*.pkl")

    all_data = {}
    chptr = orbax.checkpoint.PyTreeCheckpointer()
    for f in tqdm(files):
        d = chptr.restore(f)
        all_data = _combine(all_data, d, config, ignore)

    save_args = orbax_utils.save_args_from_target(all_data)
    chptr.save(filename, all_data, save_args=save_args)


def get_key(d):
    assert len(d.keys()) == 1
    return list(d.keys())[0]


def get_keys(d):
    return list(d.keys())


def _combine(all_data, d, config, ignore):
    key = get_key(d)
    hyper_index = hyper.index_of(config, d[key]["config"])

    seed_key = get_key(d[key]["data"])
    for ig in ignore:
        d_keys = d[key]["data"][seed_key].keys()
        if ig not in d_keys:
            msg = f"key {ig} not found in {d_keys}"
            raise KeyError(msg)
        del d[key]["data"][seed_key][ig]

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
