import os
from pprint import pprint
import orbax
from flax.training import orbax_utils
import sparseeg.util.hyper as hyper
import numpy as np
import matplotlib.pyplot as plt

chptr = orbax.checkpoint.PyTreeCheckpointer()
data_file = "./results/set_500epochs/combined/"
# Idea: use pickle to save in combine.py, maybe that would be faster?
print("Loading...")
data = chptr.restore(data_file)
print("Loaded")


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


print("Tuning")
to_tune = "valid_accuracy"
perfs = hyper.perfs(data, to_tune, inner_agg, outer_agg)
print(perfs)
b = hyper.best(perfs, np.mean)
pprint(data[str(b)]["config"])

# Get data to plot and smooth it
to_plot = "test_accuracy"
plot_data = hyper.get(data, b, to_plot)
plot_data = smooth(plot_data, 0)

n_classes = plot_data.shape[1]

# Mean +/- stderr over seeds
n_seeds = plot_data.shape[-1]
mean_plot_data = plot_data.mean(-1)
stderr_plot_data = np.std(plot_data, axis=-1) / np.sqrt(n_seeds)
n_readings = plot_data.shape[0]
x_values = np.arange(0, n_readings)
x_values = np.expand_dims(x_values, axis=1).repeat(n_classes, 1)

print(str(b), to_plot)
labels = np.array([f"Class {i}" for i in range(n_classes)])
print(plot_data.shape, labels.shape)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(x_values, mean_plot_data, label=labels)
for i in range(n_classes):
    ax.fill_between(
        x_values[:, i],
        mean_plot_data[:, i] - stderr_plot_data[:, i],
        mean_plot_data[:, i] + stderr_plot_data[:, i],
        alpha=0.25,
    )
ax.legend()

ax.set_ylim((0, 1))
ax.set_xlim((0, n_readings))
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")
title = "Dense" if "dense" in data_file else "Set"
ax.set_title(title)

save_name = "dense" if "dense" in data_file else "set"
fig.savefig(f"{os.path.expanduser('~')}/{save_name}.png")
