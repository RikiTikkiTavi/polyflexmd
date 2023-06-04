# polyflexmd

## Lammps

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
