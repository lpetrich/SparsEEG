import yaml
import pprint

def sweeps(config, index):
    out = {}
    _sweeps(config, 1, out, 1)
    return out

def _sweeps(config, index, out, accum):
    if isinstance(config, dict):
        keys = config.keys()
    elif isinstance(config, list):
        keys = range(len(config))

    for key in keys:
        print(accum)
        if isinstance(config[key], dict):
            out[key] = {}
            accum = _sweeps(config[key], index, out[key], accum)
        elif isinstance(config[key], list) and len(config[key]) > 0:
            num = len(config[key])
            out[key] = config[key][(index//accum) % num]
            print(
                    index,
                    accum,
                    num,
                    config[key],
                    config[key][(index//accum) % num],
                    out[key],
            )
            accum *= num
        else:
            num = 1
            out[key] = config[key]

    return accum


config_file = "config/dense.yaml"
with open(config_file, "r") as infile:
    config = yaml.safe_load(infile)
pprint.pprint(sweeps(config, 1))

# def sweeps(parameters, index):
#     # Get the hyperparameters corresponding to the argument index
#     out_params = {}
#     accum = 1
#     for key in parameters:

#         num = len(parameters[key])
#         out_params[key] = parameters[key][(index // accum) % num]
#         accum *= num

#     return (out_params, accum)


# def total(parameters):
#     """
#     Similar to sweeps but only returns the total number of
#     hyperparameter combinations. This number is the total number of distinct
#     hyperparameter settings. If this function returns k, then there are k
#     distinct hyperparameter settings, and indices 0 and k refer to the same
#     distinct hyperparameter setting.

#     Parameters
#     ----------
#     parameters : dict
#         The dictionary of parameters, as found in the agent's json
#         configuration file

#     Returns
#     -------
#     int
#         The number of distinct hyperparameter settings
#     """
#     return sweeps(parameters, 0)[1]


