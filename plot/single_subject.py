import scipy
import os
from pprint import pprint
import orbax
from flax.training import orbax_utils
import sparseeg.util.hyper as hyper
import numpy as np
import matplotlib.pyplot as plt
import click
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def inner_agg(x):
    return np.mean(x)


def outer_agg(x, axis):
    return np.mean(x, axis=axis)


def smooth(x, over):
    if over == 0:
        return x
    if isinstance(over, int):
        kernel = np.ones(over) / over
    else:
        kernel = over

    return np.apply_along_axis(np.convolve, 0, x, kernel, mode="valid")


def get_title(data_file):
    if "dense" in data_file:
        title = "Dense"
    elif "set" in data_file:
        title = "Set"
    elif "weight_pruning" in data_file:
        title = "Weight Pruning"
    else:
        title = "Unknown"

    return title


def get_save_file(data_file):
    if "dense" in data_file:
        save_name = "dense"
    elif "set" in data_file:
        save_name = "set"
    elif "weight_pruning" in data_file:
        save_name = "weight_pruning"
    else:
        save_name = "unknown"

    return save_name + f"_single_subject"


def get_train_percents(data):
    p = set()
    for k in data.keys():
        p.add(data[k]["config"]["train_percent"])
    return list(p)


@click.argument("data_files", type=click.Path(exists=True), nargs=-1)
@click.command
def plot(data_files):
    chptr = orbax.checkpoint.PyTreeCheckpointer()

    fig = plt.figure(figsize=(15, 5))
    axes = fig.subplots(1, len(data_files))
    all_data = []

    for k, data_file in enumerate(data_files):
        ax = axes.ravel()[k]
        print("Loading...")
        data = chptr.restore(data_file)
        print("Loaded")

        train_percent = get_train_percents(data)
        train_percent = sorted(train_percent)
        # print(train_percent)

        to_tune = "valid_accuracy"
        to_plot = "test_accuracy"
        sep_data = []
        best_hypers = []
        plot_data = []
        for p in train_percent:
            new_data = hyper.satisfies(
                data,
                lambda x: (
                    x["train_percent"] == p and
                    x["dataset"]["batch_size"] == 8192
                ),
            )
            sep_data.append(new_data)

            perfs = hyper.perfs(
                new_data, to_tune, inner_agg, outer_agg, combined=False,
            )
            h = hyper.best(perfs, np.mean)
            best_hypers.append(h)

            _plot_data = hyper.get(new_data, h, to_plot, combined=False)
            plot_data.append(_plot_data)

        plot_data = np.array(plot_data)
        all_data.append(plot_data)

        plot_data = plot_data.mean(axis=1)

        colours = ["black", "red", "green", "blue"]

        mean_plot_data = plot_data.mean(axis=-1)
        n_seeds = plot_data.shape[-1]
        std_err_plot_data = np.std(plot_data, axis=-1) / np.sqrt(n_seeds)

        # print("Mean accuracy:", mean_plot_data)
        # print("Stderr accuracy:", std_err_plot_data)

        # print(mean_plot_data.shape)
        # print(std_err_plot_data.shape)
        labels = [f"Class {c}" for c in range(4)]
        significance = 0.05
        for i in range(mean_plot_data.shape[-1]):

            ci = [[], []]
            for j in range(plot_data.shape[0]):
                conf = bs.bootstrap(
                    plot_data[j, i, :], stat_func=bs_stats.mean,
                    alpha=significance,
                )
                ci[0].append(conf.lower_bound)
                ci[1].append(conf.upper_bound)
            ci = np.array(ci)

            if k == 0:
                ax.plot(
                    train_percent, mean_plot_data[:, i], label=labels[i],
                    color=colours[i],
                )
                ax.legend()
            else:
                ax.plot(train_percent, mean_plot_data[:, i], color=colours[i])

            # ax.fill_between(
            #     train_percent,
            #     mean_plot_data[:, i] - std_err_plot_data[:, i],
            #     mean_plot_data[:, i] + std_err_plot_data[:, i],
            #     alpha=0.2, color=colours[i],
            # )
            ax.fill_between(
                train_percent, ci[0], ci[1], alpha=0.2, color=colours[i],
            )

        ax.set_xticks(train_percent)

        if k != 0:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            ax.set_ylabel("Accuracy")
            ax.set_yticks(np.arange(0.6, 1.01, 0.1))

        if k == len(data_files) // 2:
            ax.set_xlabel("Percentage of Data Used for Training")
        else:
            ax.set_xlabel("")

        ax.set_ylim(0.6, 1.0)
        alg_name = get_title(data_file)
        ax.set_title(alg_name)

        filename = (
            f"{os.path.expanduser('~')}/SparsEEGPlots/" +
            f"Separate/train_size_sensitivity_per_class"
        )

        fig.savefig(filename + ".png", bbox_inches="tight")
        fig.savefig(filename + ".svg", bbox_inches="tight")

    # print(scipy.stats.ttest_ind(all_data[0], all_data[1], axis=-1))


if __name__ == "__main__":
    plot()
