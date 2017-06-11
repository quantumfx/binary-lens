#!/bin/bash -l
#PBS -q workq
#PBS -r n
#PBS -l nodes=48:ppn=8,walltime=4:00:00
#PBS -N lensing
 
# load modules (must match modules used for compilation)
module load intel/intel-17 openmpi/2.0.1-intel-17 gcc/5.4.0 python/2.7.13
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR
 
# EXECUTION COMMAND; -np = nodes*ppn
mpirun -np 384 python lensing.py
