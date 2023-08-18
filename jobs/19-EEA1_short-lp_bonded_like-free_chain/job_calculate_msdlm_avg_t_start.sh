#!/bin/bash

#SBATCH --job-name="polyflexmd-19-EEA1_short-lp_bonded_like-free_chain-ef6e4e76-calculate_msdlm_avg_over_t_start"
#SBATCH --output="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/19-EEA1_short-lp_bonded_like-free_chain/ef6e4e76/logs/slurm-msdlm_avg_over_t_start-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/19-EEA1_short-lp_bonded_like-free_chain/ef6e4e76/logs/slurm-msdlm_avg_over_t_start-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=64:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem="120GB"

module load modenv/hiera GCC/11.3.0 OpenMPI/4.1.4 Python/3.9.6

source /beegfs/ws/0/s4610340-polyflexmd/polyflexmd/.venv/bin/activate

python -m polyflexmd.data_analysis.transform.cli calculate-msdlm-avg-over-t-start \
  /beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/19-EEA1_short-lp_bonded_like-free_chain/ef6e4e76/data/processed/lm_trajectory.csv \
  /beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/19-EEA1_short-lp_bonded_like-free_chain/ef6e4e76/data/processed/lm_msd_avg.csv \
  --style l_K \
  --take-n-first 1000 \
  --t-start 400 \
  --n-workers 8 \
  --chunk-size 5