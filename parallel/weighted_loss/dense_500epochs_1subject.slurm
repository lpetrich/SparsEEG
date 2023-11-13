#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1Gb
#SBATCH --time=01:00:00
#SBATCH --account=rrg-whitem
#SBATCH --array=0-1

module load python/3.10
source ./venv/bin/activate

srun python -m sparseeg -s "test_results" -i $SLURM_ARRAY_TASK_ID default.py eeg_low_final/non_weighted_loss/dense_500epochs_1subject.yaml
