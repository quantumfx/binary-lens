"""
Dispersive delays
"""
import numpy as np

def generate(amp = 0.5e-6):
    x = np.random.rand(150)*2-1
    y = np.random.rand(150)*2-1
    z = (1+0j)*x + (0+1j)*y
    q = np.array([0]*1851)
    z = np.concatenate((z,q))
    z = np.fft.irfft(z)
    amp = 0.5*10**(-6)
    z = z*amp/np.max(np.abs(z))
    return z
