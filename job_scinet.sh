#!/bin/bash
#PBS -l nodes=24:ppn=8,walltime=1:30:00
#PBS -N test2_caustic
 
# load modules (must match modules used for compilation)
module load intel/15.0.2 intelmpi python/2.7.8
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR
 
# EXECUTION COMMAND; -np = nodes*ppn
mpirun -np 192 python lensing.py 1> error.txt 2> out.txt
