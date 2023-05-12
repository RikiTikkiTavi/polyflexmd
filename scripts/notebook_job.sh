#!/bin/bash

#SBATCH --job-name="polyflexmd-notebook"
#SBATCH --output="/scratch/ws/0/s4610340-bt-eea1-md-workspace/slurm-%j.out"
#SBATCH --error="/scratch/ws/0/s4610340-bt-eea1-md-workspace/slurm-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=8:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=7000

module load modenv/hiera GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0

source /scratch/ws/0/s4610340-bt-eea1-md-workspace/polyflexmd/.venv/bin/activate

export XDG_RUNTIME_DIR=""

cd /scratch/ws/0/s4610340-bt-eea1-md-workspace/polyflexmd/notebooks && jupyter notebook --no-browser --port=8888
