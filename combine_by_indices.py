#!/usr/bin/env python
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

from tqdm import tqdm
import click
import pickle
import os
import yaml
import sparseeg.util.hyper as hyper
import orbax
from flax.training import orbax_utils
import fnmatch


@click.argument("inds", nargs=-1)
@click.argument("config")
@click.option("-s", "--filename", default="combined")
@click.option("-f", "--force", is_flag=True)
@click.option("-i", "--ignore")
@click.command()
def combine(inds, config, filename, force, ignore):
    if ignore is not None:
        ignore = ignore.split(",")
    config_path = config
    if not os.path.exists(config_path):
        raise Exception(f"Config file does not exist: {config_path}")

    # Parse config file
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)

    data_dir = os.path.join("./results", config["save_dir"])
    print("Combining data in", data_dir)

    filename = os.path.join(data_dir, filename)
    if os.path.exists(filename) and not force:
        raise ValueError(f"{filename} exists, use -f/--force to overwrite")

    total = hyper.total(config)
    n_seed = len(config["seed"])
    files = []
    _inds = set(map(int, inds))
    for i in range(total * n_seed):
        h = hyper.index_of(config, hyper.sweeps(config, i)[0])
        if h in _inds:
            files.append(f"{i}.pkl")

    files = list(map(lambda f: os.path.join(data_dir, f), files))

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
