#! /usr/bin/env python

# Example file call: ./schedule.py --debug clusters/3Hr_20Cores_3Seq_4Gb.json
#   ./ sparseeg "test_results" default.py dense.yaml

import yaml
import logging
import pickle
import time
import sys
import os
sys.path.append(os.getcwd() + '/src')  # noqa: E402 hack: could use PYTHONPATH
import math
from collections import OrderedDict
import numpy as np
import PyExpUtils.runner.Slurm as Slurm
import PyExpUtils.runner.parallel as Parallel
from PyExpUtils.utils.generator import group
from src.sparseeg.util.hyper import total
import click
from socket import gethostname

fewest_gpu = {
    "cedar": 4,
    "graham": 2,
}

cwd = os.getcwd()


def using_gpu(gpu):
    return (isinstance(gpu, int) and gpu > 0) or isinstance(gpu, str)


def get_fewest_gpus():
    host = gethostname()
    if "cedar" in host:
        return fewest_gpu["cedar"]
    elif "gra" in host:
        return fewest_gpu["graham"]


def getJobScript(parallel, num_gpu):
    # The contents of the string below will be the bash script that is
    # scheduled on Compute Canada. Change the script accordingly
    # (e.g. add the necessary `module load X` commands etc.)
    if isinstance(num_gpu, int) and num_gpu <= 0:
        gpus = ""
    else:
        if num_gpu == "fewest":
            num_gpu = get_fewest_gpus()
        elif not isinstance(num_gpu, int):
            raise ValueError(f"unknown gpu configuration {num_gpu}")

        gpus = f"#SBATCH --gres=gpu:{num_gpu}"

    return f"""#!/bin/bash
{gpus}

cd {cwd}
module load python/3.10
module load cuda
module load cudnn/8.0.3

source ~/py3_10/bin/activate

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
{parallel}
    """


def printProgress(size, it):
    for i, _ in enumerate(it):
        print(f'{i + 1}/{size}', end='\r')
        if i - 1 == size:
            print()
        yield _


def estimateUsage(indices, groupSize, cores, hours):
    jobs = math.ceil(len(indices) / groupSize)

    total_cores = jobs * cores
    core_hours = total_cores * hours

    core_years = core_hours / (24 * 365)
    allocation = 724

    return core_years, 100 * core_years / allocation


def gatherMissing(
    save_path, experiment_file, config_file, groupSize, cores,
    total_hours, debug,
):
    out = {}

    config_file = os.path.join("src/sparseeg/config", config_file)
    experiment_file = os.path.join("src/sparseeg/experiment", experiment_file)

    with open(config_file, "r") as infile:
        config = yaml.safe_load(infile)
    save_path = os.path.join(save_path, config["save_dir"])

    if not os.path.exists(save_path):
        to_run = sorted(range(total(config)))
        approximate_cost = estimateUsage(to_run, groupSize, cores, total_hours)
        if debug:
            print(f'{save_path}: \trunning {len(to_run)} experiments')
            print(f'hypers: \t{to_run}')
        return {save_path: to_run}, approximate_cost

    files = os.listdir(save_path)
    n_hypers = total(config)

    files = list(map(lambda x: int(x.removesuffix(".pkl")), files))
    have = set(files)
    need = set(range(n_hypers))
    to_run = sorted(list(need - have))

    out[save_path] = to_run

    approximate_cost = estimateUsage(to_run, groupSize, cores, total_hours)

    # log how many are missing
    if debug:
        print(f'{save_path}: running {len(to_run)} experiments')
        print(to_run)

    return out, approximate_cost


@click.command()
@click.argument("slurm_path")
@click.argument("base_path")
@click.argument("package")
@click.argument("save_path")
@click.argument("experiment_file")
@click.argument("config_file")
@click.option(
    "-g", "--gpu", default=0, show_default=True, type=str,
    help="how many GPUs to " +
    "request for the job. If an integer > 0, then request that many GPUs. " +
    "If an integer <= 0, then use no GPUs. If 'fewest', then request the " +
    "fewest number of GPUs available on any node in the cluster",
)
@click.option(
    "-y", "--yes", is_flag=True, show_default=True,
    help="do not confirm before scheduling",
)
@click.option(
    "-d", "--debug", is_flag=True, show_default=True,
    help="show debugging messages",
)
def main(
    slurm_path,
    base_path,
    package,
    save_path,
    experiment_file,
    config_file,
    gpu,
    yes,
    debug,
):
    # Scheduling logic
    slurm = Slurm.fromFile(slurm_path)

    # compute how many "tasks" to clump into each job
    groupSize = slurm.cores * slurm.sequential

    # compute how much time the jobs are going to take
    hours, minutes, seconds = slurm.time.split(':')
    total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

    # gather missing and sum up cost
    missing, cost = gatherMissing(
        save_path, experiment_file, config_file, groupSize, slurm.cores,
        total_hours, debug,
    )

    if not yes:
        print()
        print(
            f"Expected to use {cost[0]:.2f} core years, which is " +
            f"{cost[1]:.4f}% of our annual allocation",
        )
        input("Press Enter to confirm or ctrl+c to exit")

    for path in missing:
        # reload this because we do bad mutable things later on
        slurm = Slurm.fromFile(slurm_path)

        for g in group(missing[path], groupSize):
            tasks = list(g)

            # Convert gpus to int if possible
            if isinstance(gpu, str):
                gpu = int(gpu) if gpu.isdecimal() else gpu

            # build the executable string
            cuda_visible = ""
            if using_gpu(gpu):
                cuda_visible = "CUDA_VISIBLE_DEVICES=$(({%} - 1)); "

            runner = f"'{cuda_visible}python -m " + \
                f"{package} -s {save_path} " + "-i {}" + \
                f" {experiment_file} {config_file}'"

            if debug:
                print(runner)

            # Ensure only to request the number of CPU cores necessary
            slurm.cores = min([slurm.cores, len(tasks)])

            # generate the gnu-parallel command for dispatching to many CPUs
            # across server nodes
            parallel = Slurm.buildParallel(
                runner, tasks, {"ntasks": slurm.cores},
            )
            print(parallel)

            # Ensure not to request more GPUs than CPUs
            if isinstance(gpu, int) and gpu > slurm.cores:
                gpu = slurm.cores
            elif gpu == "fewest" and get_fewest_gpus() > slurm.cores:
                gpu = slurm.cores
            elif isinstance(gpu, int) and (gpu > 0 and gpu < slurm.cores):
                logging.warning(
                    "Fewer GPUs than CPUs requested, are you sure this " +
                    "is what you wanted to do?"
                )
            script = getJobScript(parallel, gpu)

            # Uncomment for debugging the scheduler to see what bash script
            # would have been scheduled
            if debug:
                print(script)

            Slurm.schedule(script, slurm)

            # Prevent overburdening the slurm scheduler. Be a good citizen.
            time.sleep(2)


if __name__ == "__main__":
    main()
