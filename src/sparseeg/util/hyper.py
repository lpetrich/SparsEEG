import yaml


def total(config):
    return sweeps(config, 0)[1]


def sweeps(config, index):
    out = {}
    accum = _sweeps(config, index, out, 1)

    if index > accum:
        raise IndexError(f"config index out of range ({index} > {accum})")

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
