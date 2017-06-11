from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

fmin = 311.25*10**6 #Hz
fban = 16*10**6 #Hz
freqs = [fmin + i*fban for i in range(3)]
fsample = 2*fban
Period = 1.6*10**(-3) #seconds
PulseWidth = Period/200

fref = fmin#Hz

FileName = "data/test2_parallel_"

# This generates the signal time series. It's some gaussian envelope * random values between +-0.5
def Signal(width=PulseWidth*fsample, length=int(Period*fsample), Noise=0):
    s = np.random.rand(length)-0.5
    t = np.array(range(length))
    t0 = length/2
    envelop = np.exp( -((t-t0)/width)**2 )
    return envelop*s

def PhaseArray(SignalBand, GeoPath, DisPath, Freq):
    LG = len(GeoPath)
    LD = len(DisPath)
    if LG > LD:
        DisPath = np.pad(DisPath,int((LG-LD)/2)+1,'edge')
        DisPath = DisPath[:LG]
    w = Freq + np.array(range(SignalBand))*fban/SignalBand # frequency array, of size fourier transform of signal
    PA = np.outer(w,GeoPath) - np.outer(((fref**2)/w),DisPath)  # the "phase array". it consist of a vertical array of frequencies, each of which has some paths associated.
    return PA

def PhaseFactor(PV):
    phase = np.exp(2*np.pi*(0+1j)*PV )
    return phase

def PathInt(PA): # semi-analytical
    PathInt = []
    #ImCount = []
    fRange, pathRange = np.shape(PA)
    for i in range(fRange):
        Onef = 0
        #image = 0
        IntRange, weight = FindIntRange(PA[i,:])
        for j in IntRange:
            Onef += weight[j] * PhaseFactor(PA[i,j])
            #image += 0
        PathInt += [Onef]
        #ImCount += [image]
    return np.array(PathInt)#, np.array(ImCount)

def FindIntRange(TotPath):
    width = 0.289
    dPath = np.gradient(TotPath)
    window = np.exp(-dPath**2/(2*(width)**2))
    IntRange = np.where(window > 3e-3)[0]
    return IntRange, window

def Scan(begin, end, freq):
    scan = range(begin,end)
    res = np.linspace(-1/3,1/3,3,endpoint='true')
    lensed = []
    spec = []
    for i in scan:
        dp = dispath[i-(dens*100):i+(dens*100+1)]
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

def DisPath():
    x = np.random.rand(10)*2-1
    y = np.random.rand(10)*2-1
    z = (1+0j)*x + (0+1j)*y
    q = np.array([0]*991)
    z = np.concatenate((z,q))
    z = np.fft.irfft(z)
    amp = 4*10**(-6)
    z = z*amp/np.max(np.abs(z))
    return z

dispath = DisPath()
if rank == 0:
    #dispath = DisPath()
    dispath = np.load('data/test2Dis.npy')
comm.Bcast(dispath, root=0)

#increasing density
dens = 4
temp = np.fft.rfft(dispath)
temp = np.concatenate((temp,np.zeros( (dens-1)*1000)))
temp = np.fft.irfft(temp)
temp *= dens
dispath = temp

slope = np.max(np.abs(np.gradient(dispath)))
m = 1.1*slope/(dens*200)
def GeoPath(center):
    gp = m*(np.arange(-dens*100,+dens*100+1)-center)**2
    return gp

s = Signal()
comm.Bcast(s, root=0)
s = s[24000:27000]
sf = np.fft.rfft(s)
l = len(sf)

gp = GeoPath(0)
PA = PhaseArray(l,gp,gp*0,fmin)
PI = PathInt(PA)
s1 = np.fft.irfft(sf*PI)
s2 = sf*PI
norm = (s1**2).sum()

if rank == 0:
    np.save(FileName+"Geo",gp)
    np.save(FileName+"Dis",dispath)
    np.save(FileName+"Freqs",freqs)
    np.save(FileName+"UnlensedSpec",s2)

if rank == 0:
    scantemp = np.arange(len(gp)-1, len(dispath) - (len(gp)-1), (len(dispath)-2*(len(gp)-1)) // ( size//len(freqs) ) )
    np.save(FileName+"Scan",scantemp)
    diff = scantemp[1] - scantemp[0]
    print scantemp, diff
    scan = []

    # useful for MPI scattering routine
    for i in range(len(freqs)):
        scan = np.append(scan,scantemp)
    print scan
else:
    scan = None
    diff = None
scan = comm.scatter(scan, root=0)
diff = comm.bcast(diff, root=0)

scan = int(scan)
diff = int(diff)

# each processor will only process diff points for one particular frequency
mag, spec = Scan(scan,scan+diff,freqs[rank%len(freqs)])
np.save(FileName + format(freqs[rank%3]/10**6, '.2f') + "Mag"+format(scan, '04') + "to" + format(scan+diff, '04'),mag)
#np.save(FileName + format(freqs[rank%3]/10**6, '.2f') + "Spec"+format(scan, '04') + "to" + format(scan+diff, '04'),spec)