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
    # k_inner : inner cutoff scale (at large k)
    # k_outer : outer cutoff scale (at small k, doesn't really matter too much, needs to be small to maintain power law)

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

def test():
    np.random.seed(0)
    test_field = generate_power_law(fs = 10, nrf = 1, amp = 1, alpha = 1, k_outer = 1e-5, k_inner = 1)
    expected = np.array([0.8283278907397528, 0.14841512324238842, -0.2107772347532253, 0.048524374029767586, -0.10693544818931458, -0.13751405176832399, -0.06765225325959, -0.12474477219417185, -0.47332512506274416, 0.09568149721546117])
    assert np.sum(np.abs(test_field - expected)) < 1e-10, 'Bug in gaussian field generation'
