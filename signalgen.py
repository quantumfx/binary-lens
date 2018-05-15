"""
Generates a signal
"""
import numpy as np

Period = 1.6e-3 #s
PulseWidth = Period/200
fban = 16e6 #Hz
fsample = 2*fban

def Signal(width=PulseWidth*fsample, length=int(Period*fsample), Noise=0):
    s = np.random.rand(length)-0.5
    t = np.array(range(length))
    t0 = length/2
    envelop = np.exp( -((t-t0)/width)**2 )
    return envelop*s
