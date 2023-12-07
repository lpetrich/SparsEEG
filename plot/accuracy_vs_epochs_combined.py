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


@click.argument("data_files", nargs=-1)
@click.argument("plot_title", nargs=1)
@click.option("-s", "--smooth_over", type=int, default=0)
@click.option("-k", "--skip", type=int, default=0)
@click.command
def plot(data_files, plot_title, smooth_over, skip):
    fig = plt.figure()
    ax = fig.add_subplot()

    colours = ["black", "red", "blue"]
    labels = ["Dense", "SET", "Weight Pruning"]

    for i, data_file in enumerate(data_files):
        chptr = orbax.checkpoint.PyTreeCheckpointer()

        print(f"Loading {data_file}...")
        data = chptr.restore(data_file)
        data = hyper.satisfies(
            data, lambda x: x["dataset"]["batch_size"] == 8192,
        )
        print("Loaded")

        print("Tuning")
        to_tune = "valid_accuracy"
        perfs = hyper.perfs(
            data, to_tune, inner_agg, outer_agg, combined=True,
        )
        print(perfs)
        b = hyper.best(perfs, np.mean)
        pprint(data[str(b)]["config"])
        print("Best:", b)

        # Get data to plot and smooth it
        test_accuracy = hyper.get(data, b, "test_accuracy", combined=True)
        train_accuracy = hyper.get(
            data, b, "train_accuracy", combined=True,
        )
        plot_data = test_accuracy  # - train_accuracy
        if skip > 0:
            plot_data = plot_data[::skip, :, :]
        plot_data = smooth(plot_data, smooth_over)

        if "dense" in data_file:
            plot_data = plot_data[::5, :]

        n_classes = plot_data.shape[1]

        # Mean +/- stderr over seeds
        n_seeds = plot_data.shape[-1]
        mean_plot_data = plot_data.mean(-1)
        stderr_plot_data = np.std(plot_data, axis=-1) / np.sqrt(n_seeds)
        n_readings = plot_data.shape[0]
        x_values = np.arange(0, n_readings)
        x_values = np.expand_dims(x_values, axis=1).repeat(n_classes, 1)

        significance = 0.05

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
            x_values[:, 1], mean_plot_data, label=labels[i],
            color=colours[i],
        )
        # ax.fill_between(
        #     x_values[:, 1],
        #     mean_plot_data - stderr_plot_data,
        #     mean_plot_data + stderr_plot_data,
        #     alpha=0.25,
        # )
        ax.fill_between(
            x_values[:, 1], ci[0], ci[1], alpha=0.2, color=colours[i],
        )

    ax.legend(loc="lower right", fontsize=22)
    # ax.get_legend().remove()

    ax.set_xlim(1, 100)
    ax.set_ylim((0.2, 1.2))
    # ax.set_xlim((0, n_readings))
    ax.set_ylabel("Accuracy", fontsize=24)
    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_xticks(range(0, 101, 20), range(0, 501, 100), fontsize=16)
    ax.set_yticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[0.2, 0.4, 0.6, 0.8, 1.0],
        fontsize=16,
    )
    ax.set_title("")

    # ax.set_yticks(())
    # ax.set_ylabel("  ")

    save_name = plot_title.lower().replace(" ", "_")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(
            f"{os.path.expanduser('~')}/SparsEEGPlots/{save_name}.{ext}",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    plot()
