import numpy as np
from scipy.special import erf
from ..psf2d import *

def jac2mix(adu,X,Y,theta,sigma=0.55,N0=1000,ntheta=2):
    nspots = len(theta) // ntheta
    mu = np.zeros_like(adu,dtype=np.float32)
    nx,ny = adu.shape
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny))
    for n in range(nspots):
        x0,y0 = theta[ntheta*n:ntheta*(n+1)]
        lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
        mu += N0*lam
    jac = 1 - adu/mu
    return jac.flatten()


