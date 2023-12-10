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

from pprint import pprint
from copy import deepcopy
import os
import fnmatch
import yaml
from copy import deepcopy
import orbax
from flax.training import orbax_utils
import numpy as np
import warnings


def satisfies(data, f):
    out_keys = []
    for hyper in data:
        if f(data[hyper]["config"]):
            out_keys.append(int(hyper))

    out = {}
    out_keys = sorted(out_keys)
    for i, k in enumerate(out_keys):
        out[str(i)] = deepcopy(data[str(k)])

    return out


def get(data, hyper, key, combined):
    hyper = str(hyper)

    subkey = "combined" if combined else "label-by-label"

    seed_data = []
    new_data = data[hyper]["data"]
    for seed in new_data:
        seed_data.append(new_data[seed][key][subkey])

    out = np.array(seed_data)
    if out.ndim == 1:
        out = np.expand_dims(out, 0)

    return out.T


def best(perfs, agg, reverse=False):
    perfs = agg(perfs, axis=1)
    if reverse:
        return np.argmin(perfs)
    else:
        return np.argmax(perfs)


def perfs(data, tune_by, inner_agg, outer_agg, combined):
    hyper_perfs = [[] for _ in range(len(data.keys()))]
    keys = list(map(lambda x: int(x), data.keys()))
    keys = list(map(lambda x: str(x), sorted(keys)))
    for j, hyper in enumerate(keys):
        hyper = str(hyper)
        seed_data = []
        for seed in data[hyper]["data"]:
            if combined:
                d = data[hyper]["data"][seed][tune_by]["combined"]
            else:
                d = data[hyper]["data"][seed][tune_by]["label-by-label"]
            label_data = []
            for i, label in enumerate(d):
                label_data.append(inner_agg(label))
            seed_data.append(label_data)

        seed_data = np.array(seed_data)

        if seed_data.ndim == 1:
            seed_data = np.expand_dims(seed_data, 0)
        # hyper_perfs[int(hyper)] = outer_agg(seed_data, axis=0)
        hyper_perfs[j] = outer_agg(seed_data, axis=0)

    out = np.array(hyper_perfs)

    # This is terrible, but oh well
    if "accuracy" in tune_by or "std" in tune_by:
        out[np.isnan(out)] = 0.0
    else:
        out[np.isnan(out)] = np.finfo(np.float32).min

    return out


def renumber(data, indices):
    indices = sorted(indices)
    new_d = {}
    new_config = deepcopy(data[indices[0]]["config"])

    for (j, i) in enumerate(indices):
        new_d[j] = data[i]
        new_config.update(data[i]["config"])

    return new_d


def renumber_to(data, src_dest):
    assert len(src_dest) == len(data)

    new_d = {}
    for k, v in src_dest:
        new_d[v] = data[k]

    return new_d


def index_of(config, setting, ignore=["seed"]):
    # Ignore seeds when numbering hypers
    setting = deepcopy(setting)
    config = deepcopy(config)
    for i in ignore:
        del setting[i]
        del config[i]

    n = total(config)
    inds = []
    for i in range(n):
        if sweeps(config, i)[0] == setting:
            return i
    return None


def total(config):
    config = deepcopy(config)
    if "seed" in config.keys():
        del config["seed"]
    return sweeps(config, 0)[1]


def sweeps(config, index):
    out = {}
    accum = _sweeps(config, index, out, 1)

    if index >= accum:
        raise IndexError(f"config index out of range ({index} >= {accum})")

    return out, accum


def _sweeps(config, index, out, accum):
    if isinstance(config, dict):
        keys = config.keys()
    elif isinstance(config, list):
        keys = range(len(config))

    for key in keys:
        if isinstance(config[key], dict):
            out[key] = {}
            accum = _sweeps(config[key], index, out[key], accum)
        elif isinstance(config[key], list) and len(config[key]) > 0:
            num = len(config[key])
            out[key] = config[key][(index // accum) % num]
            accum *= num
        else:
            num = 1
            out[key] = config[key]

    return accum
