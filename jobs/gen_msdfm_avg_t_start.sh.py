import pathlib

job = """#!/bin/bash

#SBATCH --job-name="polyflexmd-{name}-msdfm_avg_over_t_start-ef6e4e76"
#SBATCH --output="{path}/logs/slurm-msdfm_avg_over_t_start-%j.out"
#SBATCH --error="{path}/logs/slurm-msdfm_avg_over_t_start-%j.out"
#SBATCH --account="p_mdpolymer"
#SBATCH --time=64:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

module load modenv/hiera GCC/11.3.0 OpenMPI/4.1.4 Python/3.9.6

source /beegfs/ws/0/s4610340-polyflexmd/polyflexmd/.venv/bin/activate

python -m polyflexmd.data_analysis.transform.cli calculate-msdlm-avg-over-t-start \\
  {path}/data/processed/fm_trajectory.csv \\
  {path}/data/processed/fm_msd_avg.csv \\
  --style {style} \\
  --take-n-first 1000 \\
  --n-workers 16 \\
  --chunk-size 5
"""

experiments = [
    {
        "path": "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/17-EEA1_short-free_chain/538accb2",
        "style": "l_K",
        "name": "17-EEA1_short-free_chain"
    },
    {
        "path": "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K-vary-d_end/18-EEA1_short+Rab5_10x-free_chain/ef6e4e76",
        "style": "l_K+d_end",
        "name": "18-EEA1_short+Rab5_10x-free_chain"
    },
    {
        "path": "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K/19-EEA1_short-lp_bonded_like-free_chain/ef6e4e76",
        "style": "l_K",
        "name": "19-EEA1_short-lp_bonded_like-free_chain"
    },
    {
        "path": "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-vary-l_K-vary-d_end/22-EEA1_short+Rab5_10x-lp_bonded_like-free_chain/ef6e4e76",
        "style": "l_K+d_end",
        "name": "22-EEA1_short+Rab5_10x-lp_bonded_like-free_chain"
    },
    {
        "path": "/beegfs/ws/0/s4610340-polyflexmd/data/experiment_results/FENE-beadspring-free_chain-full_flex/20-full_flex-free_chain/ef6e4e76",
        "style": "simple",
        "name": "20-full_flex-free_chain"
    },

]

for experiment in experiments:
    path_current = pathlib.Path(__file__).parent.resolve()
    path_job = path_current / experiment["name"] / "job_calculate_fm_msd_avg_t_start.sh"
    with open(path_job, "w") as file:
        file.write(job.format(**experiment))