#!/bin/bash

#SBATCH --job-name="polyflexmd-17-calculate_fm_msd-ef6e4e76"
#SBATCH --output="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/17-EEA1_short-free_chain/ef6e4e76/logs/calculate_fm_msd-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/17-EEA1_short-free_chain/ef6e4e76/logs/calculate_fm_msd-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=64:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

module load modenv/hiera GCC/11.3.0 OpenMPI/4.1.4 Python/3.9.6

source /beegfs/ws/0/s4610340-polyflexmd/polyflexmd/.venv/bin/activate

EXPERIMENT_PATH = "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/17-EEA1_short-free_chain/ef6e4e76"

python -m polyflexmd.data_analysis.transform.cli calculate-msd \
  "${EXPERIMENT_PATH}/data/processed/fm_trajectory.csv" \
  --style l_K \
  --output-path ${EXPERIMENT_PATH}/data/processed/fm_msd.csv