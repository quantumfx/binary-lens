"""
Path integration
"""
import numpy as np

import phase
import geopath

def PathInt(PA):
    PathInt = []
    fRange, pathRange = np.shape(PA)
    for i in range(fRange):
        Onef = 0
        IntRange, weight = FindIntRange(PA[i,:])
        for j in IntRange:
            Onef += weight[j] * phase.PhaseFactor(PA[i,j])
        PathInt += [Onef]
    return np.array(PathInt)

# Window function for path integration
def FindIntRange(TotPath):
    width = 0.289
    dPath = np.gradient(TotPath)
    window = np.exp(-dPath**2/(2*(width)**2)) # window function = weight of each path
    IntRange = np.where(window > 3e-3)[0]
    return IntRange, window

def Scan(begin, end, freq):
    scan = range(begin,end)
    #res = np.linspace(-1/2,1/2,3,endpoint='true')[:-1] # divides each point into          subpoints of source position
    res = np.array([0])
    lensed = []
    spec = []
    for i in scan:
        dp = dispath[i-(dens*500):i+(dens*500+1)]
        for j in res:
            gp = geopath.generate(j)
            PA = phase.PhaseArray(l,gp,dp,freq)
            PI = PathInt(PA)
            s1 = np.fft.irfft(sf*PI)
            # save intensity as well as spectrum
            lensed += [(s1**2).sum()]
            spec += [sf*PI]
        print(freq/10**6, i, begin, end, time.clock())
    return np.array(lensed)/norm, np.array(spec)
