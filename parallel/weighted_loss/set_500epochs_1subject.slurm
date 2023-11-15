#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2Gb
#SBATCH --time=06:00:00
#SBATCH --account=def-whitem
#SBATCH -e err/%x_%j_%a.err
#SBATCH -o out/%x_%j_%a.out
#SBATCH --array=1,2,3,4,5,18,19,20,21,23,37,38,39,40,41,54,55,56,57,59,72,73,74,75,76,77,90,92,93,94,95

module load python/3.10
source ./venv/bin/activate

RESULTS_DIR="results"
EXPERIMENT_FILE="default.py"
CONFIG_FILE="eeg_low_final/weighted_loss/set_500epochs_1subject.yaml"

srun python -m sparseeg -s $RESULTS_DIR -i $SLURM_ARRAY_TASK_ID $EXPERIMENT_FILE $CONFIG_FILE
