import os
from pprint import pprint
import orbax
from flax.training import orbax_utils
import sparseeg.util.hyper as hyper
import numpy as np
import matplotlib.pyplot as plt
import click


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

    return save_name


@click.argument("data_file", type=click.Path(exists=True))
@click.option("-s", "--smooth_over", type=int, default=0)
@click.option("-c", "--combined", is_flag=True, default=False)
@click.option("-k", "--skip", type=int, default=0)
@click.command
def plot(data_file, smooth_over, combined, skip):
    chptr = orbax.checkpoint.PyTreeCheckpointer()

    print("Loading...")
    data = chptr.restore(data_file)
    # data = hyper.satisfies(data, lambda x: x["dataset"]["batch_size"] == 0)
    print("Loaded")

    print("Tuning")
    to_tune = "valid_accuracy"
    perfs = hyper.perfs(data, to_tune, inner_agg, outer_agg, combined=combined)
    print(perfs)
    b = hyper.best(perfs, np.mean)
    pprint(data[str(b)]["config"])
    print("Best:", b)

    # Get data to plot and smooth it
    to_plot = "test_accuracy"
    plot_data = hyper.get(data, b, to_plot, combined=combined)
    if skip > 0:
        plot_data = plot_data[::skip, :, :]
    plot_data = smooth(plot_data, smooth_over)

    n_classes = plot_data.shape[1]

    # Mean +/- stderr over seeds
    n_seeds = plot_data.shape[-1]
    mean_plot_data = plot_data.mean(-1)
    stderr_plot_data = np.std(plot_data, axis=-1) / np.sqrt(n_seeds)
    n_readings = plot_data.shape[0]
    x_values = np.arange(0, n_readings)
    x_values = np.expand_dims(x_values, axis=1).repeat(n_classes, 1)

    fig = plt.figure()
    ax = fig.add_subplot()

    if not combined:
        labels = np.array([f"Class {i}" for i in range(n_classes)])
        ax.plot(x_values, mean_plot_data, label=labels)
        for i in range(n_classes):
            ax.fill_between(
                x_values[:, i],
                mean_plot_data[:, i] - stderr_plot_data[:, i],
                mean_plot_data[:, i] + stderr_plot_data[:, i],
                alpha=0.25,
            )
    else:
        labels = np.array(["Accuracy"])
        ax.plot(x_values[:, 1], mean_plot_data, label=labels)
        ax.fill_between(
            x_values[:, 1],
            mean_plot_data - stderr_plot_data,
            mean_plot_data + stderr_plot_data,
            alpha=0.25,
        )
    ax.legend()

    ax.set_ylim((-0.05, 1.05))
    # ax.set_xlim((0, n_readings))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    title = "Dense" if "dense" in data_file else "Set"
    ax.set_title(get_title(data_file))

    save_name = get_save_file(data_file)

    fig.savefig(f"{os.path.expanduser('~')}/{save_name}.png")


if __name__ == "__main__":
    plot()
