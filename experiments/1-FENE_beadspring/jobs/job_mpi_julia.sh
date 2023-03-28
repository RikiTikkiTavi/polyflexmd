#!/bin/bash

#SBATCH --job-name="FENE-beadspring-test"
#SBATCH --output="/home/h2/s4610340/Projects/bachelor-thesis/experiments/FENE_beadspring/logs/slurm-%j.out"
#SBATCH --error="/home/h2/s4610340/Projects/bachelor-thesis/experiments/FENE_beadspring/logs/slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=24:00:00
#SBATCH --partition=julia
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000M

module load modenv/hiera GCC/9.3.0 iccifort/2020.1.217 OpenMPI/4.0.3

mpirun singularity exec \
  --contain \
  --no-home \
  -B experiments/FENE_beadspring:/experiments/FENE_beadspring \
  --env OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  docker://lammps/lammps:stable_29Sep2021_ubuntu20.04_openmpi_py3 \
  lmp_mpi -in /experiments/FENE_beadspring/lammps-input/in.lammps -log /experiments/FENE_beadspring/logs/log.txt
