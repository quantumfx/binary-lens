from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import time
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

fmin = 311.25*10**6 #Hz
fban = 16*10**6 #Hz
nfreq = int(sys.argv[1]) # which frequency band above 311.25 to scan
freq = fmin + nfreq * fban
fsample = 2*fban
Period = 1.6*10**(-3) #seconds
PulseWidth = Period/200
#R = 60 #s
R = 6.273 #s
fref = fmin #Hz

# conversion factors between time and points
intime = 32/8000 # second/point

# Output prefix
FileName = "data/constrained_0.5Hz_freqcut_"

# Generates signal
def Signal(width=PulseWidth*fsample, length=int(Period*fsample), Noise=0):
    s = np.random.rand(length)-0.5
    t = np.array(range(length))
    t0 = length/2
    envelop = np.exp( -((t-t0)/width)**2 )
    return envelop*s

# Phase function phi(f,x)
def PhaseArray(SignalBand, GeoPath, DisPath, Freq):
    LG = len(GeoPath)
    LD = len(DisPath)
    if LG > LD:
        DisPath = np.pad(DisPath,int((LG-LD)/2)+1,'edge')
        DisPath = DisPath[:LG]
    w = Freq + np.array(range(SignalBand))*fban/SignalBand # frequency array, of size fourier transform of signal
    PA = np.outer(w,GeoPath) - np.outer(((fref**2)/w),DisPath)
    return PA

def PhaseFactor(PV):
    phase = np.exp(2*np.pi*(0+1j)*PV )
    return phase

def PathInt(PA):
    PathInt = []
    fRange, pathRange = np.shape(PA)
    for i in range(fRange):
        Onef = 0
        IntRange, weight = FindIntRange(PA[i,:])
        for j in IntRange:
            Onef += weight[j] * PhaseFactor(PA[i,j])
        PathInt += [Onef]
    return np.array(PathInt)

# Window function for path integration
def FindIntRange(TotPath):
    width = 0.289
    dPath = np.gradient(TotPath)
    window = np.exp(-dPath**2/(2*(width)**2)) # window function = weight of each path
    IntRange = np.where(window > 3e-3)[0]
    return IntRange, window

#
def Scan(begin, end, freq):
    scan = range(begin,end)
    res = np.linspace(-1/2,1/2,3,endpoint='true')[:-1] # divides each point into subpoints of source position
    lensed = []
    spec = []
    for i in scan:
        dp = dispath[i-(dens*500):i+(dens*500+1)]
        for j in res:
            gp = GeoPath(j)
            PA = PhaseArray(l,gp,dp,freq)
            PI = PathInt(PA)
            s1 = np.fft.irfft(sf*PI)
            # save intensity as well as spectrum
            lensed += [(s1**2).sum()]
            spec += [sf*PI]
        print freq/10**6, i, begin, end, time.clock()
    return np.array(lensed)/norm, np.array(spec)

# Gaussian DM
def DisPath():
    x = np.random.rand(150)*2-1
    y = np.random.rand(150)*2-1
    z = (1+0j)*x + (0+1j)*y
    q = np.array([0]*1851)
    z = np.concatenate((z,q))
    z = np.fft.irfft(z)
    amp = 0.5*10**(-6)
    z = z*amp/np.max(np.abs(z))
    return z

dispath = np.empty( 4*len(DisPath()) )
if rank == 0:
    #dispath = DisPath()
    dispath = np.load('data/constrained_0.5Hz.npy')
comm.Bcast(dispath, root=0)

#increasing density, do this if you're generating a new dispath
dens = 4
#temp = np.fft.rfft(dispath)
#temp = np.concatenate((temp,np.zeros( (dens-1)*2000)))
#temp = np.fft.irfft(temp)
#temp *= dens
#dispath = temp

# artificially generates a geometric delay
# slope = np.max(np.abs(np.gradient(dispath)))
# m = 1.1*slope/(dens*200)
# def GeoPath(center):
#     gp = m*(np.arange(-dens*100,+dens*100+1)-center)**2
#     return gp

# Physical parameters for geometric delay
def GeoPath(center):
    gp = 1/(2*R) * ((np.arange(-dens*500,+dens*500+1)-center) * intime *358e3/3e8 )**2 # 358e3 m/s is binary relative velocity, 3e8 m/s is speed of light
    return gp

#s = np.empty( len(Signal()) )
#if rank == 0:
#    s = Signal()
#comm.Bcast(s, root=0)
#s = s[24000:27000]
# if comparing different runs, should spectrum/magnification normalize by the same signal
s = np.load('data/signal.npy')
sf = np.fft.rfft(s)
l = len(sf)

gp = GeoPath(0)
PA = PhaseArray(l,gp,gp*0,fmin)
PI = PathInt(PA)
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
    print scan, diff
else:
    scan = None
    diff = None
scan = comm.scatter(scan, root=0)
diff = comm.bcast(diff, root=0)

scan = int(scan)
diff = int(diff)

# each processor will only process diff points for one particular frequency
mag, spec = Scan(scan,scan+diff,freq)
# comm.Barrier()

# print "process", rank, "has", mag

# if rank == 0 :
#     magGathered = np.zeros(len(mag) * size)
# else:
#     magGathered = None

#comm.Gatherv(mag, [magGathered, np.ones(size)*len(mag), np.arange(size)*len(mag), MPI.DOUBLE])

#np.save(FileName + format(freq/10**6, '.2f') + "Mag", magGathered)
np.save(FileName + format(freq/10**6, '.2f') + "Mag"+format(scan, '05') + "to" + format(scan+diff, '05'),mag)
np.save(FileName + format(freq/10**6, '.2f') + "Spec"+format(scan, '05') + "to" + format(scan+diff, '05'),spec)
