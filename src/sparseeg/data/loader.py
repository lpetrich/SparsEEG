"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License. 
"""
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from sparseeg.data.dataset import *
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold


####################################################################
# Taken from https://jax.readthedocs.io/en/latest/notebooks/
# Neural_Network_and_Data_Loading.html#data-loading-with-pytorch
####################################################################
def numpy_collate(batch):
    return jax.tree_util.tree_map(np.asarray, data.default_collate(batch))


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
    if isinstance(ds, WineDataset):
        return NumpyLoader(ds, *args, **kwargs)
    if isinstance(ds, WAYEEGGALDataset):
        return NumpyLoader(ds, *args, **kwargs)
    else:
        raise NotImplementedError(
            f"No DataLoader implemented for Dataset of type {type(ds)}",
        )
