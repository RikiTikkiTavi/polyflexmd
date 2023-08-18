#!/bin/bash

#SBATCH --job-name="polyflexmd-18-EEA1_short+Rab5_10x-free_chain-process-ef6e4e76"
#SBATCH --output="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K-vary-d_end/18-EEA1_short+Rab5_10x-free_chain/ef6e4e76/logs/slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K-vary-d_end/18-EEA1_short+Rab5_10x-free_chain/ef6e4e76/logs/slurm-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=64:00:00
#SBATCH --partition=haswell
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10583

module load modenv/hiera GCC/11.3.0 OpenMPI/4.1.4 Python/3.9.6

source /beegfs/ws/0/s4610340-polyflexmd/polyflexmd/.venv/bin/activate

python -m polyflexmd.data_analysis.pipelines process-experiment-data /beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K-vary-d_end/18-EEA1_short+Rab5_10x-free_chain/ef6e4e76 --style l_K+d_end --no-l-k-estimate --time-steps-per-partition 100 --partition "haswell" --cores 2 --memory "20GB" --max-workers 10