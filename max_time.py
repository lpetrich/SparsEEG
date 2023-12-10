#!/usr/bin/env python
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

from tqdm import tqdm
from datetime import timedelta
import click
import orbax
from flax.training import orbax_utils
import os


@click.argument("dir")
@click.command()
def max_time(dir):
    if not dir.endswith("combined"):
        dir = os.path.join(dir, "combined")

    chptr = orbax.checkpoint.PyTreeCheckpointer()
    data = chptr.restore(dir)

    max_time = 0
    for hyper in tqdm(data):
        d = data[hyper]["data"]
        for seed in d:
            max_time = max(max_time, d[seed]["total_time"])

    t = timedelta(seconds=max_time)
    print(f"{t} HH:MM:SS")


if __name__ == "__main__":
    max_time()
