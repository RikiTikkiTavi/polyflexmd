#!/bin/bash

docker run \
  -v experiments/FENE_beadspring:/home/lammps/experiments/FENE_beadspring \
  --entrypoint /usr/bin/lmp_serial \
  --env OMP_NUM_THREADS=4 \
  lammps/lammps -in /home/lammps/experiments/FENE_beadspring/lammps-input/in.lammps
