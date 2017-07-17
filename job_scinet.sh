#!/bin/bash
#PBS -l nodes=30:ppn=8,walltime=2:00:00
#PBS -N lensing
 
# load modules (must match modules used for compilation)
module load intel/15.0.2 intelmpi python/2.7.8
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR
 
# EXECUTION COMMAND; -np = nodes*ppn
# Frequency band is 311.25MHz to 311.25MHZ+16MHz*NFREQ, where NFREQ is the number after "lensing.py"
mpirun -np 240 python "lensing.py" 0 #1> out.txt 2> error.txt
