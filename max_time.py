#!/usr/bin/env python


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
