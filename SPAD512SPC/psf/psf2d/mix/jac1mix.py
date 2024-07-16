import numpy as np
from ..psf2d import jac1

def jac1mix(x,y,theta,ntheta=2):
    nspots = len(theta) // ntheta
    jacblock = [jac1(x,y,theta[ntheta*n:ntheta*(n+1)]) for n in range(nspots)]
    return np.concatenate(jacblock)
