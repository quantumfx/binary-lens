#!/bin/bash
#SBATCH --nodes=6
#SBATCH --ntasks=240
#SBATCH --time=2:00:00
#SBATCH --job-name lensing
 
# load modules (must match modules used for compilation)
module purge
module load gcc/7.3.0 python/3.6.5 openmpi/3.1.0
#module load intel intelmpi python/2.7.14
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd /scratch/p/pen/flin/lensing-sim-data/
 
# EXECUTION COMMAND; -np = nodes*ppn
# Frequency band is 311.25MHz+16MHz*NFREQ to 311.25MHZ+16MHz*(NFREQ+1), where NFREQ is the number after "lensing.py"
mpirun -np 160 python3 "~/gpc-scratch/binary-lens/lensing.py" 9 #1> out.txt 2> error.txt
