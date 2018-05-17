"""
Dispersive delays

generate_new generates a dispersive phase with N points and fs px/Fresnel scale. Should have about 100px/Fresnel scale (ie. should have ~100px < 2pi in the geometric phase).

Generating delays instead of phase so it can be scaled easily later.
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

def generate_power_law(fs = 100, nrf = 100, amp = 5e-7, alpha = -2.0, k_outer = 1e-5, k_inner = None):
    # fs      : px/Fresnel scale
    # nrf     : number of Fresnel scales to simulate
    # amp     : delay fluctuation amplitude
    # alpha   : power law exponent
    # k_inner : inner cutoff scale
    # k_outer : outer cutoff scale (doesn't really matter too much, needs to be small to maintain power law)

    nx = nrf*fs #number of points
    x  = np.linspace(0, nrf, nrf * fs)
    kx = np.fft.rfftfreq(nrf * fs, 1 / fs)

    # compute power spectrum
    psd = amp**2 * (kx**2 + k_outer**2)**(-alpha / 2)
    if k_inner != None:
        cutoff = np.exp( -(kx/(2*k_inner) )**2 )
        psd   *= cutoff

    dk_r = np.random.normal(scale = np.sqrt(psd) )
    dk_i = np.random.normal(scale = np.sqrt(psd) )

    dk = dk_r + 1j*dk_i

    # hack to set mean to 0
    dk[0] = 0

    delay = np.fft.irfft(dk, norm = 'ortho')

    return delay
