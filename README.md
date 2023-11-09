# polyflexmd

This repository contains scripts used for the research
of dynamical properties of semiflexible polymer chains
using the molecular dynamics methods within my bachelor thesis
"[Molecular dynamics study of ideal
polymer chains with variable persistence
length](thesis/out/BA.pdf)".

## Structure

- `src/` - python source code of data analysis and utils for simulations setup
    - `src/data_analysis` - data analysis
    - `src/experiment_runner` - utility to setup lammps simulation jobs
    - `src/system_creator` - utility to create an initial system for the simulation
- `experiments` - contains the experiment definition files (configs)
- `simulations` - LAMMPS simulation definition files
- `notebooks` - jupyter notebooks
- `reports` - reports generated from jupyter notebooks
- `jobs` - data processing job definitions for several experiments
- `deploy` - makefile to build LAMMPS for the used hardware
- `thesis` - thesis

## Workflow

Experiment definition files (`./experiments`) are meant to be the complete
definition of the experiment workflow.
They define necessary parameters to setup the simulation.
An `experiment_runner` module through the CLI interface
accepts the experiment definition file
and creates necessary folders and slurm job files to execute the simulation.
The data produced by the simulation is then processed either in jupyter notebooks
or firstly aggregated using the data analysis pipelines and then analyzed in jupyter notebooks.

## Running experiments

Following instruction applies to execute the experiments and execute data analysis.
Python ">=3.9,<3.12" and poetry are required.

1. Create venv: `python -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Install dependencies: `poetry install`
4. Run experiment using `experiment_runner` CLI: `experiment-runner run {path-to-experiment-def-file}`

## Build LAMMPS

Lammps version used: `stable_23Jun2022_update4`

Taurus lmod modules used: `modenv/hiera GCC/11.3.0 OpenMPI/4.1.4 Python/3.9.6`

### Build and install

1. `git clone git@github.com:lammps/lammps.git --branch stable_23Jun2022_update4`
2. `cd lammps/src`
3. `cp {path-to-repo}/deploy/lammps/make/Makefile.omp_romeo_opt ./MAKE/MINE/Makefile.omp_romeo_opt`
4. `module load modenv/hiera GCC/11.3.0 OpenMPI/4.1.4`
5. `make -j 16 yes-molecule mode=shared omp_romeo_opt`
6. `source {path-to-repo}/.venv/bin/activate`
7. `make install-python`

6-7 steps are only necessary if you want to use lammps from python, which is not the case here.