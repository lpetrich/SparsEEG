import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from data.dataset import *
import torch.utils.data as data


####################################################################
# Taken from https://jax.readthedocs.io/en/latest/notebooks/
# Neural_Network_and_Data_Loading.html#data-loading-with-pytorch
####################################################################
def numpy_collate(batch):
    return jax.tree_util.tree_map(
        np.asarray, data.default_collate(batch),
    )


class NumpyLoader(DataLoader):
    def __init__(
        self, dataset, batch_size, shuffle, sampler=None,
        batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            collate_fn=numpy_collate, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
####################################################################


def setup(ds, *args, **kwargs):
    if isinstance(ds, ClassifierDataset):
        return NumpyLoader(ds, *args, **kwargs)
    else:
        raise NotImplementedError(
            f"No DataLoader implemented for Dataset of type {type(ds)}",
        )
