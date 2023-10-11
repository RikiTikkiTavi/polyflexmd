#!/bin/bash

#SBATCH --job-name="polyflexmd-20-full_flex-free_chain-calculate_fm_traj-ef6e4e76"
#SBATCH --output="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-full_flex/20-full_flex-free_chain/ef6e4e76/logs/calculate_fm_traj-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-full_flex/20-full_flex-free_chain/ef6e4e76/logs/calculate_fm_traj-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=64:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G

module load modenv/hiera GCC/11.3.0 OpenMPI/4.1.4 Python/3.9.6

source /beegfs/ws/0/s4610340-polyflexmd/polyflexmd/.venv/bin/activate

python -m polyflexmd.data_analysis.transform.cli extract-fm-trajectory \
  "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-full_flex/20-full_flex-free_chain/ef6e4e76/data/processed/trajectories/*.csv" \
  --style "simple" \
  --output-path /beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-full_flex/20-full_flex-free_chain/ef6e4e76/data/processed/fm_trajectory.csv \
  --partition "romeo" \
  --cores 6 \
  --memory "60GB" \
  --max-workers 10