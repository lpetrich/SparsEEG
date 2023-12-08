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

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    all_data = []

    for k, data_file in enumerate(data_files):
        print("Loading...")
        data = chptr.restore(data_file)
        print("Loaded")

        train_percent = get_train_percents(data)
        train_percent = sorted(train_percent)
        # print(train_percent)

        to_tune = "valid_accuracy"
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
                new_data, to_tune, inner_agg, outer_agg, combined=True,
            )
            h = hyper.best(perfs, np.mean)
            best_hypers.append(h)

            print(p)
            if "set" in data_file:
                kwargs = new_data[str(h)]["config"]["model"]["optim"]["kwargs"]
                print("\t", kwargs["drop_fraction_fn"])
                print("\t", kwargs["scheduler"]["kwargs"]["update_freq"])
                print("\t", kwargs["sparsity_distribution_fn"]["kwargs"])
                kwargs = new_data[str(h)]["config"]["model"]["optim"]
                print("\t", kwargs["wrapped"]["kwargs"]["learning_rate"])
            if "weight_pruning" in data_file:
                kwargs = new_data[str(h)]["config"]["model"]["optim"]["kwargs"]
                print("\t", kwargs["scheduler"]["args"])
                print("\t", kwargs["sparsity_distribution_fn"]["kwargs"])
                kwargs = new_data[str(h)]["config"]["model"]["optim"]
                print("\t", kwargs["wrapped"]["kwargs"]["learning_rate"])


            test_accuracy = hyper.get(
                new_data, h, "test_accuracy", combined=True,
            )
            train_accuracy = hyper.get(
                new_data, h, "train_accuracy", combined=True,
            )
            _plot_data = test_accuracy  # - train_accuracy
            plot_data.append(_plot_data)

        plot_data = np.array(plot_data)
        all_data.append(plot_data)

        plot_data = plot_data.mean(axis=1)

        colours = ["black", "red", "blue"]

        mean_plot_data = plot_data.mean(axis=-1)
        n_seeds = plot_data.shape[-1]
        std_err_plot_data = np.std(plot_data, axis=-1) / np.sqrt(n_seeds)

        ci = [[], []]
        significance = 0.05
        for j in range(plot_data.shape[0]):
            conf = bs.bootstrap(
                plot_data[j, :], stat_func=bs_stats.mean,
                alpha=significance,
            )
            ci[0].append(conf.lower_bound)
            ci[1].append(conf.upper_bound)
        ci = np.array(ci)

        # print("Mean accuracy:", mean_plot_data)
        # print("Stderr accuracy:", std_err_plot_data)

        # print(mean_plot_data.shape)
        # print(std_err_plot_data.shape)
        labels = ["Dense", "SET", "Weight Pruning"]
        ax.plot(
            train_percent, mean_plot_data, label=labels[k], color=colours[k],
        )
        # ax.fill_between(
        #     train_percent, mean_plot_data - std_err_plot_data,
        #     mean_plot_data + std_err_plot_data, alpha=0.2, color=colours[k],
        # )
        ax.fill_between(
            train_percent, ci[0], ci[1], alpha=0.2, color=colours[k],
        )

    ax.set_xticks(train_percent)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_xlabel("Percentage of Data Used for Training", fontsize=16)
    ax.set_yticks(np.arange(0.7, 1.01, 0.1))
    ax.set_ylim(0.7, 1.0)
    # ax.set_ylim(-0.20, 0.0)
    # ax.set_yticks(np.arange(-0.2, 0.01, 0.05))
    ax.set_title("", fontsize=24)
    ax.legend(loc="lower right", fontsize=12)

    filename = (
        f"{os.path.expanduser('~')}/SparsEEGPlots/" +
        f"Separate/train_size_sensitivity_combined_classes"
    )

    fig.savefig(filename + ".png", bbox_inches="tight")
    fig.savefig(filename + ".svg", bbox_inches="tight")


if __name__ == "__main__":
    plot()
