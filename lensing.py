import numpy as np
import time
from mpi4py import MPI
import sys

#local modules
import signalgen
import phase
import pathint
import dispath
import geopath

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

fmin = 311.25e6 #Hz
fban = 16e6 #Hz
nfreq = int(sys.argv[1]) # which frequency band above 311.25 to scan
freq = fmin + nfreq * fban
#fsample = 2*fban
#Period = 1.6*10**(-3) #seconds
#PulseWidth = Period/200
#R = 6.273 #c*s
fref = fmin #Hz

# conversion factors between time and points
#intime = 32/8000 # second/point

# Output prefix
FileName = "data/corrugated_sheet_"

# Generates signal

# Phase function phi(f,x)

# Path Int

# Path int scan

# Gaussian DM

# dispath

#dispath = np.empty( 4*len(dispath.generate()) )
dispath = np.zeros(20000)
if rank == 0:
    #dispath = dispath.generate()
    dispath = np.load('data/corrugated_sheet.npy')
comm.Bcast(dispath, root=0)

#increasing density, do this if you're generating a new dispath
dens = 4
#temp = np.fft.rfft(dispath)
#temp = np.concatenate((temp,np.zeros( (dens-1)*2000)))
#temp = np.fft.irfft(temp)
#temp *= dens
#dispath = temp


#s = np.empty( len(Signal()) )
#if rank == 0:
#    s = Signal()
#comm.Bcast(s, root=0)
#s = s[24000:27000]
# if comparing different runs, should spectrum/magnification normalize by the same signal
s = np.load('data/signal.npy')
sf = np.fft.rfft(s)
l = len(sf)

gp = geopath.generate(0)
PA = phase.PhaseArray(l,gp,gp*0,fmin)
PI = pathint.PathInt(PA)
s1 = np.fft.irfft(sf*PI)
norm = (s1**2).sum()

if rank == 0:
    np.save(FileName+"Geo",gp)
    np.save(FileName+"Dis",dispath)
    np.save(FileName+"Signal",s)
    np.save(FileName+"UnlensedSpec",sf*PI)

# number of points to scan through should be divisible by number of cores
if rank == 0:
    scan = np.arange((len(gp)-1)/2, len(dispath) - (len(gp)-1)/2, (len(dispath)-(len(gp)-1)) // size )
    np.save(FileName+"Scan",scan)
    diff = scan[1] - scan[0]
    print(scan, diff)
else:
    scan = None
    diff = None
scan = comm.scatter(scan, root=0)
diff = comm.bcast(diff, root=0)

scan = int(scan)
diff = int(diff)

# each processor will only process diff points for one particular frequency
mag, spec = pathint.Scan(scan,scan+diff,freq)
# comm.Barrier()

# print( "process", rank, "has", mag)

# if rank == 0 :
#     magGathered = np.zeros(len(mag) * size)
# else:
#     magGathered = None

#comm.Gatherv(mag, [magGathered, np.ones(size)*len(mag), np.arange(size)*len(mag), MPI.DOUBLE])

#np.save(FileName + format(freq/10**6, '.2f') + "Mag", magGathered)
np.save(FileName + format(freq/10**6, '.2f') + "Mag"+format(scan, '05') + "to" + format(scan+diff, '05'),mag)
np.save(FileName + format(freq/10**6, '.2f') + "Spec"+format(scan, '05') + "to" + format(scan+diff, '05'),spec)
