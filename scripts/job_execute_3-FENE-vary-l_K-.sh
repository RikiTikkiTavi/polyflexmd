#!/bin/bash

#SBATCH --job-name="polyflexmd-3-FENE-vary-l_K"
#SBATCH --output="/scratch/ws/0/s4610340-bt-eea1-md-workspace/data/experiment_results/FENE-beadspring-vary-l_K/4-FENE-beadspring-vary-l_K/e64ff746/logs/slurm-%j.out"
#SBATCH --error="/scratch/ws/0/s4610340-bt-eea1-md-workspace/data/experiment_results/FENE-beadspring-vary-l_K/4-FENE-beadspring-vary-l_K/e64ff746/logs/slurm-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=24:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=15500

module load modenv/hiera GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0

source /scratch/ws/0/s4610340-bt-eea1-md-workspace/polyflexmd/.venv/bin/activate

papermill /scratch/ws/0/s4610340-bt-eea1-md-workspace/polyflexmd/notebooks/3-FENE-vary-l_K.ipynb /scratch/ws/0/s4610340-bt-eea1-md-workspace/data/experiment_results/FENE-beadspring-vary-l_K/4-FENE-beadspring-vary-l_K/e64ff746/3-FENE-vary-l_K.ipynb --kernel polyflexmd -p PATH_EXPERIMENT "/scratch/ws/0/s4610340-bt-eea1-md-workspace/data/experiment_results/FENE-beadspring-vary-l_K/4-FENE-beadspring-vary-l_K/e64ff746"

jupyter nbconvert --to=HTML --output="3-FENE-vary-l_K" --output-dir="/scratch/ws/0/s4610340-bt-eea1-md-workspace/data/experiment_results/FENE-beadspring-vary-l_K/4-FENE-beadspring-vary-l_K/e64ff746" "/scratch/ws/0/s4610340-bt-eea1-md-workspace/data/experiment_results/FENE-beadspring-vary-l_K/4-FENE-beadspring-vary-l_K/e64ff746/3-FENE-vary-l_K.ipynb"