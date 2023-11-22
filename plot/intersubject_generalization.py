import pandas as pd
from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
import jax.numpy as jnp
from sparseeg.experiment import default as default
from flax.training import train_state
from sparseeg.training import state as training
import jax
import flax.linen as nn
import sparseeg.data.dataset as dataset
import sparseeg.data.loader as loader
from sparseeg.approximator.mlp import MLP
import os
from pprint import pprint
import orbax
from flax.training import orbax_utils
import sparseeg.util.hyper as hyper
import numpy as np
import matplotlib.pyplot as plt
import click


# We'll need to average over seeds eventually
i = 2
inds = [3, 4, 5]
subjects = [
    [1, 2, 3, 4, 5, 6, 10],
    [1, 2, 3, 4, 7, 8, 9],
    [1, 4, 5, 6, 7, 8, 10],
]
data_file = f"./results/dense_500epochs_3subject/{inds[i]}.pkl"
chptr = orbax.checkpoint.PyTreeCheckpointer()

print("Loading...")
data = chptr.restore(data_file)
print("Loaded")

key = list(data.keys())[0]
seed = list(data[key]["data"].keys())[0]
model_state = data[key]["data"][seed]["model"]

ds_config = {
    "type": "WAYEEGGALDataset-Low",
    "batch_size": None,
    "shuffle": False,
    "subjects": subjects[i],
    # "subjects": [2, 3],  # seed 1
    "percent": 1.0,

}

ds = dataset.WAYEEGGALDataset("low", ds_config, 1)
batch_size = 30000
dl = loader.NumpyLoader(ds, batch_size, ds_config["shuffle"])

labels = []
predictions = []

model = default.model_fn(data[key]["config"]["model"], 1, ds)

# Compute label metrics
for x_batch, y_batch in dl:
    batch = {"inputs": x_batch, "labels": y_batch}
    logits = model.apply(
        {'params': model_state["params"]}, batch['inputs'],
    )
    pred = jnp.argmax(logits, axis=-1)

    labels.extend(y_batch)
    predictions.extend(pred)

labels = np.array(labels).ravel()
predictions = np.array(predictions).ravel()

print(type(labels), labels.shape)
print(type(predictions), predictions.shape)

pp_matrix_from_data(labels, predictions, cmap="BuPu")
print(skmetrics.precision_recall_fscore_support(
    labels, predictions
))


#     fig = plt.figure()
#     axes = fig.subplots(2, 2)
#     conf = skmetrics.multilabel_confusion_matrix(y_batch, predicted)
#     for i in range(len(axes.ravel())):
#         disp = skmetrics.ConfusionMatrixDisplay(
#             conf[i, :, :],
#         )
#         disp.plot()

#         axes.ravel()[i] = disp.ax_

#         fig.savefig("/home/samuel/conf.png")
