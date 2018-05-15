"""
Geometric delay
"""
import numpy as np

#1957
R      = 6.273 # c*s
intime = 32/8000 # seconds/pixel

#corrugated sheet
au     = 1.495978707e11   #meters
c      = 2.99792458e8     #meters
pc     = 3.0857e16        #meters
re     = 2.8179403227e-15 #meters
dlens  = 389.*pc
dpsr   = 620.*pc
beta   = 1. - dlens/dpsr
deff   = beta*dlens
freq   = 311.25e6
lo     = c/freq
rf     = np.sqrt(deff*lo)
T      = 0.03*au*10
dne1   = 0.003*1e6/10

# artificially generates a geometric delay
# slope = np.max(np.abs(np.gradient(dispath)))
# m = 1.1*slope/(dens*200)
# def GeoPath(center):
#     gp = m*(np.arange(-dens*100,+dens*100+1)-center)**2
#     return gp

# Physical parameters for geometric delay
# def generate(center):
#     gp = 1/(2*R) * ((np.arange(-dens*500,+dens*500+1)-center) * intime *358e3/3e8 )**2 # 358e3 m/s is binary relative velocity, 3e8 m/s is speed of light
#     return gp

def generate(center):
    gp = (T*(np.linspace(-3,3,num=6001,endpoint=True)-center*0.001))**2/(2*rf**2*freq)
    return gp
