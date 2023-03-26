#/bin/bash

docker run \
 -v /home/egor/Projects/bachelor-thesis/experiments/FENE_beadspring:/home/lammps/experiments/FENE_beadspring \
 --entrypoint /usr/bin/lmp_mpi \
 --env OMP_NUM_THREADS=8 \
 lammps/lammps -in /home/lammps/experiments/FENE_beadspring/in.lammps