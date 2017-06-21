#!/bin/bash
#PBS -l nodes=1:ppn=8,walltime=4:00:00
#PBS -N test2_caustic
 
# load modules (must match modules used for compilation)
module load intel/15.0.2 intelmpi python/2.7.8
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR
 
# EXECUTION COMMAND; -np = nodes*ppn
# Frequency band is 311.25MHz + 16MHz * NFREQ
mpirun -np 8 python "randomcaustic_11-16.py" 4 #1> out.txt 2> error.txt
