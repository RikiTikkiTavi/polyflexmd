#!/bin/bash

#SBATCH --job-name="polyflexmd-notebook"
#SBATCH --output="/beegfs/ws/0/s4610340-polyflexmd/.logs/slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-polyflexmd/.logs/slurm-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=8:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=7000

module load modenv/hiera GCC/11.3.0 OpenMPI/4.1.4 Python/3.9.6

source /beegfs/ws/0/s4610340-polyflexmd/polyflexmd/.venv/bin/activate

export XDG_RUNTIME_DIR=""

cd /beegfs/ws/0/s4610340-polyflexmd/polyflexmd/notebooks && jupyter notebook --no-browser --port=8888
