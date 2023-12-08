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

    smoothed = np.apply_along_axis(np.convolve, 0, x, kernel, mode="valid")
    return np.concatenate((x[0:1, :, :], smoothed), axis=0)


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


def get_save_file(data_file, combined):
    if "dense" in data_file:
        save_name = "dense"
    elif "set" in data_file:
        save_name = "set"
    elif "weight_pruning" in data_file:
        save_name = "weight_pruning"
    else:
        save_name = "unknown"

    n = "".join(filter(str.isdigit, data_file))[-1]

    return save_name + f"_{n}" + ("_combined" if combined else "")


@click.argument("data_file", type=click.Path(exists=True))
@click.option("-s", "--smooth_over", type=int, default=0)
@click.option("-c", "--combined", is_flag=True, default=False)
@click.option("-k", "--skip", type=int, default=0)
@click.command
def plot(data_file, smooth_over, combined, skip):
    chptr = orbax.checkpoint.PyTreeCheckpointer()

    print("Loading...")
    data = chptr.restore(data_file)
    data = hyper.satisfies(data, lambda x: x["dataset"]["batch_size"] == 8192)
    print("Loaded")

    print("Tuning")
    to_tune = "valid_accuracy"
    perfs = hyper.perfs(data, to_tune, inner_agg, outer_agg, combined=combined)
    print(perfs)
    b = hyper.best(perfs, np.mean)
    pprint(data[str(b)]["config"])
    print("Best:", b)

    # Get data to plot and smooth it
    test_accuracy = hyper.get(data, b, "test_accuracy", combined=combined)
    # train_accuracy = hyper.get(data, b, "train_accuracy", combined=combined)
    plot_data = test_accuracy
    if skip > 0:
        plot_data = plot_data[::skip, :, :]
    plot_data = smooth(plot_data, smooth_over)

    plot_data = plot_data[::5, :, :]

    n_classes = plot_data.shape[1]

    # Mean +/- stderr over seeds
    n_seeds = plot_data.shape[-1]
    mean_plot_data = plot_data.mean(-1)
    stderr_plot_data = np.std(plot_data, axis=-1) / np.sqrt(n_seeds)
    n_readings = plot_data.shape[0]
    x_values = np.arange(0, n_readings)
    x_values = np.expand_dims(x_values, axis=1).repeat(n_classes, 1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot()
    significance = 0.05

    colours = ["black", "red", "blue", "green"]
    if not combined:
        labels = np.array([f"Class {i}" for i in range(n_classes)])
        for i in range(n_classes):

            ci = [[], []]
            for j in range(plot_data.shape[0]):
                conf = bs.bootstrap(
                    plot_data[j, i, :], stat_func=bs_stats.mean,
                    alpha=significance,
                )
                ci[0].append(conf.lower_bound)
                ci[1].append(conf.upper_bound)
            ci = np.array(ci)
            ax.plot(
                x_values[:, i], mean_plot_data[:, i], label=labels[i],
                color=colours[i],
            )
            ax.fill_between(
                x_values[:, i], ci[0], ci[1], alpha=0.2,
                color=colours[i],
            )

    else:
        ci = [[], []]
        for j in range(plot_data.shape[0]):
            conf = bs.bootstrap(
                plot_data[j, :], stat_func=bs_stats.mean,
                alpha=significance,
            )
            ci[0].append(conf.lower_bound)
            ci[1].append(conf.upper_bound)
        ci = np.array(ci)

        ax.plot(
            x_values[:, 1], mean_plot_data, label="Accuracy",
            color="black",
        )
        # ax.fill_between(
        #     x_values[:, 1],
        #     mean_plot_data - stderr_plot_data,
        #     mean_plot_data + stderr_plot_data, alpha=0.25,
        # )
        ax.fill_between(
            x_values[:, 1], ci[0], ci[1], alpha=0.2, color="black",
        )
    ax.legend(loc="lower right", fontsize=12)

    ax.set_ylim((-0.05, 1.2))
    # ax.set_xlim((0, n_readings))
    ax.set_ylabel("Accuracy", fontsize=22)
    ax.set_xlabel("Epochs", fontsize=22)

    ax.set_xticks(
        range(0, 101, 20),
        labels=range(0, 501, 100),
        fontsize=16,
    )
    ax.set_yticks(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        fontsize=16,
    )

    save_name = get_save_file(data_file, combined)

    fig.savefig(
        f"{os.path.expanduser('~')}/SparsEEGPlots/{save_name}_minibatch.png",
        bbox_inches="tight",
    )
    fig.savefig(
        f"{os.path.expanduser('~')}/SparsEEGPlots/{save_name}_minibatch.svg",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    plot()
