import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
from scipy.special import erf
from ..psf2d import *

def mixloglike(theta,adu,nparams=2,sigma=0.55,N0=1000):
    nx,ny = adu.shape
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny))
    mu = np.zeros_like(adu,dtype=np.float32)
    nspots = len(theta)//nparams
    for n in range(nspots):
        x0,y0 = theta[nparams*n:nparams*(n+1)]
        lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
        mu += N0*lam
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    stirling = adu * np.nan_to_num(np.log(adu)) - adu
    p = adu*np.log(mu)
    warnings.filterwarnings("default", category=RuntimeWarning)
    p = np.nan_to_num(p)
    nll = stirling + mu - p
    nll = np.sum(nll)
    return nll
