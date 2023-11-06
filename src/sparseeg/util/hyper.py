import yaml
from copy import deepcopy
import orbax
from flax.training import orbax_utils
import numpy as np


def best(perfs, agg, reverse=False):
    perfs = agg(perfs, axis=1)
    if reverse:
        return np.argmin(perfs)
    else:
        return np.argmax(perfs)


def perfs(data, tune_by, inner_agg, outer_agg):
    hyper_perfs = [[] for _ in range(len(data.keys()))]
    print(len(hyper_perfs), data.keys())
    for hyper in data:
        if hyper not in "1":
            break
        seed_data = []
        for seed in data[hyper]["data"]:
            d = data[hyper]["data"][seed][tune_by]
            label_data = []
            for i, label in enumerate(d):
                print(label, i, seed, hyper)
                label_data.append(inner_agg(label))
                print()
            seed_data.append(label_data)

        seed_data = np.array(seed_data)

        if seed_data.ndim == 1:
            seed_data = np.expand_dims(seed_data, 0)
        hyper_perfs[int(hyper)] = outer_agg(seed_data, axis=0)

    out = np.array(hyper_perfs)
    out[np.isnan(out)] = np.finfo(np.float32).min

    return out.T  # (n_hypers × labels)


def renumber(data, indices):
    indices = sorted(indices)
    new_d = {}
    new_config = deepcopy(data[indices[0]]["config"])

    for (j, i) in enumerate(indices):
        new_d[j] = data[i]
        new_config.update(data[i]["config"])

    return new_d


def satisfies(data, config, f):
    new_config = deepcopy(config)

    # Clear the config, just keeping the structure
    for key in new_config:
        if isinstance(new_config[key], list):
            new_config[key] = set()

    indices = []
    for index in data.keys():
        hypers = data[index]["config"]
        if not f(hypers):
            continue

        # Track the hyper indices and full hyper settings
        indices.append(index)

    return indices


def index_of(config, setting, ignore=["seed"]):
    # Ignore seeds when numbering hypers
    setting = deepcopy(setting)
    config = deepcopy(config)
    for i in ignore:
        del setting[i]
        del config[i]

    n = total(config)
    for i in range(n):
        if sweeps(config, i)[0] == setting:
            return i
    return None


def total(config):
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
