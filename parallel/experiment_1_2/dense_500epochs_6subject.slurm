#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6Gb
#SBATCH --time=12:00:00
#SBATCH --account=def-whitem
#SBATCH -e err/%x_%j_%a.err
#SBATCH -o out/%x_%j_%a.out
#SBATCH --array=0-20

module load python/3.10
source ./venv/bin/activate

RESULTS_DIR="results"
EXPERIMENT_FILE="default.py"
CONFIG_FILE="dense_500epochs_6subject.yaml"

srun python -m sparseeg -s $RESULTS_DIR -i $SLURM_ARRAY_TASK_ID $EXPERIMENT_FILE $CONFIG_FILE
