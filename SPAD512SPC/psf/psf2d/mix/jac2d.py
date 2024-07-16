import numpy as np
from .jac1mix import *
from .jac2mix import *
from .mll2d import *

def jacmix(theta,adu,ntheta=2):
    nx,ny = adu.shape
    nspots = len(theta) // ntheta
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny))
    J1 = jac1mix(X,Y,theta)
    J1 = J1.reshape((ntheta*nspots,nx**2))
    J2 = jac2mix(adu,X,Y,theta)
    J = J1 @ J2
    J = J.reshape((ntheta,nspots))
    return J


