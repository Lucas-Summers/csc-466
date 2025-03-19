#!/bin/bash
#SBATCH --job-name=python_array_job
#SBATCH --output=outfiles/python_array_%A_%a.out
#SBATCH --error=outfiles/python_array_%A_%a.err
#SBATCH --array=0-836
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Add your private modules directory and load your conda environment
module use $HOME/privatemodules
module load conda-env/466-py3.8.8

# since arrays are limited from 0-1000, we need to run this script 3 times
# to get the full range of 0-2836
start_index=$((SLURM_ARRAY_TASK_ID + 2000))

echo $start_index
# Run the Python script with the array task ID as the parameter for n
python colab_filter.py $start_index 1000
