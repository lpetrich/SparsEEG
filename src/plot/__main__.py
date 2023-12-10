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

from collections.abc import MutableMapping
import click
from pprint import pprint
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pickle
from sparseeg.util import hyper as hyper
import os
import yaml
import sparseeg.util.hyper as hyper
from cv_evaluate.util import cv_evaluate
from tqdm import tqdm


@click.command()
@click.argument("dir")
@click.argument("config")
@click.argument("plot_over")
@click.option("-f", "--filename", default="data.pkl")
@click.option(
    "-s", "--savefile", default=f"{os.path.expanduser('~')}/plot.png",
)
@click.option(
    "-m", "--argmin", is_flag=True, help="use an argmin to " +
    "optimize over hypers instead of an argmax"
)
@click.option(
    "-c", "--cache-datasets", is_flag=True, default=False,
    help="cache the train, validation, and testing datasets for further " +
    "analysis offline",
)
def plot(dir, config, plot_over, filename, savefile, argmin, cache_datasets):
    fig = plt.figure(figsize=(10, 20))
    axs = fig.subplots(5, 2)
    for j, alg in enumerate(("SET", "Dense")):
        # alg = "SET"
        d = f"{alg.lower()}_wine"
        if alg == "SET":
            with open("./src/sparseeg/config/set_wine.yaml", "r") as i:
                config = yaml.safe_load(i)
        else:
            with open("./src/sparseeg/config/dense_wine.yaml", "r") as i:
                config = yaml.safe_load(i)
        tune_over = "test_accuracy"
        metric = "accuracy"
        type_ = "generalization gap"
        for i, net_size in tqdm(enumerate((16, 32, 64, 128, 256))):
            f = os.path.join("./results", d, f"data_{net_size}.pkl")

            ax = axs[i, j]
            ax.set_ylim(-0.1, 0.1)

            if alg != "Dense":
                continue
            if net_size != 16:
                continue

            if alg == "Dense" and net_size == 256:
                ax.set_ylim(0, 1)
                ax.plot([0, 250], np.arange(0, 2), color="red")
                ax.plot([0, 250], np.arange(1, -1, -1), color="red")
                ax.set_xlabel("Epochs")
                fig.savefig(f"/home/samuel/{fname}.png", bbox_inches="tight")
                continue

            with open(f, "rb") as infile:
                data = pickle.load(infile)

            if type_ == "test accuracy":
                test = cv_evaluate(
                    data, config, tune_over, f"test_{metric}", argmin,
                )
                data = test  # - train  # - train
            elif type_ == "train accuracy":
                train = cv_evaluate(
                    data, config, tune_over, f"train_{metric}", argmin,
                )
                data = train
            elif type_ == "generalization gap":
                train = cv_evaluate(
                    data, config, tune_over, f"train_{metric}", argmin,
                )
                test = cv_evaluate(
                    data, config, tune_over, f"test_{metric}", argmin,
                )
                data = test - train
                print(data.mean())
            else:
                raise NotImplementedError()

            size = 10
            kernel = np.ones(size) / size
            data = np.apply_along_axis(
                np.convolve, -1, data, kernel, mode="valid",
            )

            n_classes = data.shape[1]
            n_runs = data.shape[0]
            epochs = data.shape[-1]

            mean = data.mean(axis=0)
            stderr = np.std(data, axis=0) / np.sqrt(n_runs)

            ax.plot(mean.T)
            # ax.fill_between(range(
            # epochs), mean-stderr, mean+stderr, alpha=0.3)
            fill_between = mean.T - stderr.T, mean.T + stderr.T
            for k in range(n_classes):
                ax.fill_between(
                    range(epochs),
                    fill_between[0][:, k], fill_between[1][:, k],
                    alpha=0.3,
                )

            if i == 0:
                print("HERE")
                ax.set_title(f"{type_.title()} ({alg})")
            if i == 4:
                ax.set_xlabel("Epochs")
            if j == 0:
                ax.set_ylabel(f"Accuracy ({net_size})")

            fname = type_.replace(" ", "_").lower()
            fig.savefig(f"/home/samuel/{fname}.png", bbox_inches="tight")


#     config_name = config
#     fpath = "./src/sparseeg/config"
#     config_path = os.path.join(fpath, config)
#     if not os.path.exists(config_path):
#         raise Exception(f"Config file does not exist: {config_path}")

#     # Parse config file
#     with open(config_path, "r") as infile:
#         config = yaml.safe_load(infile)

#     filename = os.path.join(dir, config["save_dir"], filename)
#     if not os.path.exists(filename):
#         raise ValueError(f"cannot find {filename}")

#     with open(filename, "rb") as infile:
#         data = pickle.load(infile)

#     tune_over = "test_accuracy"
#     metric = "accuracy"
#     # print(data[0]["data"]["seed_0"]["external"]["fold_0"].keys())
#     test = cv_evaluate(data, config, tune_over, f"test_{metric}", argmin)
#     train = cv_evaluate(data, config, tune_over, f"train_{metric}", argmin)

#     type_ = "test"
#     if type_ == "test":
#         data = test  # - train  # - train
#     elif type_ == "train":
#         data = train
#     elif type_ == "gap":
#         data = test - train
#     else:
#         raise NotImplementedError()

#     size = 10
#     kernel = np.ones(size) / size
#     data = np.apply_along_axis(np.convolve, -1, data, kernel, mode="valid")
#     # data = data.mean(1)

#     n_classes = data.shape[1]
#     n_runs = data.shape[0]
#     epochs = data.shape[-1]

#     mean = data.mean(axis=0)
#     stderr = np.std(data, axis=0) / np.sqrt(n_runs)

#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.plot(mean.T)
#     # ax.fill_between(range(epochs), mean-stderr, mean+stderr, alpha=0.3)
#     fill_between = mean.T - stderr.T, mean.T + stderr.T
#     for i in range(n_classes):
#         ax.fill_between(
#             range(epochs), fill_between[0][:, i], fill_between[1][:, i],
#             alpha=0.3,
#         )

#     # ax.set_ylim((0.0, 1.0))
#     # ax.set_ylim((-0.08, 0.02))

#     if "set" in config_name.lower():
#         title = "SET"
#     else:
#         title = "Dense"

#     ax.set_title(f"{type_} Accuracy ({title})")
#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("Accuracy")

    # fig.savefig(savefile)
    # fig.savefig(f"/home/samuel/plot.png")


if __name__ == "__main__":
    plot()
