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
