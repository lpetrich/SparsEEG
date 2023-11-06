import os
from pprint import pprint
import orbax
from flax.training import orbax_utils
import sparseeg.util.hyper as hyper
import numpy as np
import matplotlib.pyplot as plt

chptr = orbax.checkpoint.PyTreeCheckpointer()
data = chptr.restore("./results/dense_large_1000epochs_full/combined")


def inner_agg(x):
    return np.mean(x)


def outer_agg(x, axis):
    return np.mean(x, axis=axis)


def smooth(x, over):
    if isinstance(over, int):
        kernel = np.ones(over) / over
    else:
        kernel = over

    return np.apply_along_axis(np.convolve, 0, x, kernel, mode="valid")


perfs = hyper.perfs(data, "valid_accuracy", inner_agg, outer_agg)
print(perfs)
b = hyper.best(perfs, np.mean)
pprint(data[str(b)]["config"])

plot_data = hyper.get(data, b, "test_accuracy")
print("SHAPE:", plot_data.shape)
print("SHAPE:", smooth(plot_data, 10).shape)
plot_data = smooth(plot_data, 10)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(plot_data.mean(axis=1))
fig.savefig(f"{os.path.expanduser('~')}/fig.png")

# for i in range(len(perfs)):
#     print(perfs[i])
#     pprint(data[str(i)]["config"])
#     print()
#     print()
