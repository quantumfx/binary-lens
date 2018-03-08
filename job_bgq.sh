#!/bin/sh
# @ job_name           = B1957_lensing
# @ job_type           = bluegene
# @ comment            = "Lensing sim"
# @ error              = $(job_name).$(Host).$(jobid).err
# @ output             = $(job_name).$(Host).$(jobid).out
# @ bg_size            = 64
# @ wall_clock_limit   = 30:00
# @ bg_connectivity    = Torus
# @ queue 

# Launch all BGQ jobs using runjob
runjob --np 1000 --ranks-per-node=16 --envs OMP_NUM_THREADS=1 HOME=$HOME LD_LIBRARY_PATH=$LD_LIBRARY_PATH PYTHONPATH=$PYTHONPATH --cwd=$SCRATCH/binary-lens/ : /scratch/s/scinet/nolta/venv-numpy/bin/python $SCRATCH/binary-lens/lensing.py 0
