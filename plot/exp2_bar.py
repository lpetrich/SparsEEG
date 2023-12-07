import seaborn as sns
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

random_1 = (.25, (.25, .25))
random_3 = (.25, (.25, .25))
random_6 = (.25, (.25, .25))
random_9 = (.25, (.25, .25))
random_err_low = [
    random_1[1][0], random_3[1][0], random_6[1][0], random_9[1][0],
],
random_err_high = [
    random_1[1][1], random_3[1][1], random_6[1][1], random_9[1][1],
],
random_height = np.array([random_1[0], random_3[0], random_6[0], random_9[0]])

dense_1 = (.328, (.317, .338))
dense_3 = (.354, (.347, .362))
dense_6 = (.356, (.351, .362))
dense_9 = (.354, (.351, .356))
dense_err_low = [dense_1[1][0], dense_3[1][0], dense_6[1][0], dense_9[1][0]],
dense_err_high = [dense_1[1][1], dense_3[1][1], dense_6[1][1], dense_9[1][1]],
dense_height = np.array([dense_1[0], dense_3[0], dense_6[0], dense_9[0]])

set_1 = (.329, (.318, .340))
set_3 = (.348, (.342, .354))
set_6 = (.352, (.348, .357))
set_9 = (.353, (.351, .354))
set_err_low = [set_1[1][0], set_3[1][0], set_6[1][0], set_9[1][0]],
set_err_high = [set_1[1][1], set_3[1][1], set_6[1][1], set_9[1][1]],
set_height = np.array([set_1[0], set_3[0], set_6[0], set_9[0]])

wp_1 = (.327, (.317, .337))
wp_3 = (.351, (.345, .358))
wp_6 = (.354, (.346, .360))
wp_9 = (.356, (.350, .359))
wp_err_low = [wp_1[1][0], wp_3[1][0], wp_6[1][0], wp_9[1][0]],
wp_err_high = [wp_1[1][1], wp_3[1][1], wp_6[1][1], wp_9[1][1]],
wp_height = np.array([wp_1[0], wp_3[0], wp_6[0], wp_9[0]])

f = plt.figure(figsize=(12, 3))
ax = f.add_subplot()
bars = ax.bar(
    x=[1, 6, 11, 16],
    height=[dense_1[0], dense_3[0], dense_6[0], dense_9[0]],
    yerr=abs(
        np.array([dense_err_low, dense_err_high])[:, 0, :] - dense_height,
    ),
    color="dimgray",
    edgecolor="black",
    linewidth=1,
)

ax.bar(
    x=[2, 7, 12, 17],
    height=[set_1[0], set_3[0], set_6[0], set_9[0]],
    yerr=abs(
        np.array([set_err_low, set_err_high])[:, 0, :] - set_height,
    ),
    color="red",
    edgecolor="black",
    linewidth=1,
)

ax.bar(
    x=[3, 8, 13, 18],
    height=[wp_1[0], wp_3[0], wp_6[0], wp_9[0]],
    yerr=abs(
        np.array([wp_err_low, wp_err_high])[:, 0, :] - wp_height,
    ),
    color="blue",
    edgecolor="black",
    linewidth=1,
)

ax.bar(
    x=[4, 9, 14, 19],
    height=[random_1[0], random_3[0], random_6[0], random_9[0]],
    yerr=abs(
        np.array([random_err_low, random_err_high])[:, 0, :] - random_height,
    ),
    color="green",
    edgecolor="black",
    linewidth=1,
)

ax.set_xticks(
    [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19],
    labels=[
        "Dense", "SET", "WP", "Random",
        "Dense", "SET", "WP", "Random",
        "Dense", "SET", "WP", "Random",
        "Dense", "SET", "WP", "Random",
    ],
    fontsize=16,
)
ax.set_xlabel("Algorithm", fontsize=22, labelpad=20)

ax.set_yticks(
    [0.0, 0.1, 0.3, 0.4],
    labels=[0.0, 0.1, 0.3, 0.4],
    fontsize=16,
)
ax.set_ylabel("Accuracy", fontsize=22)

plt.xticks(rotation=90)


f.savefig(
    "/home/samuel/SparsEEGPlots/intersubj_generalization.png",
    bbox_inches="tight",
)
f.savefig(
    "/home/samuel/SparsEEGPlots/intersubj_generalization.svg",
    bbox_inches="tight",
)
