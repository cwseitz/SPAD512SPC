import numpy as np
from scipy.ndimage import median_filter
from skimage.feature import blob_log

def compute_sigma(G20,G2m,numm,B):
    dG20 = np.sqrt(G20)
    dG2m = np.sqrt(G2m/numm)
    dB = np.sqrt(B)
    dg20dB = (G20-G2m)/(G2m-B)**2
    dg20dG2m = (G20-B)/(G2m-B)**2
    dg20dG20 = 1/(G2m-B)
    sigma = np.array([dg20dB*dB,dg20dG2m*dG2m,dg20dG20*dG20])
    sigma = np.linalg.norm(sigma)
    return sigma

def coincidence_ratio(counts,B=0.0):
    ms = np.arange(1,100,1)
    G20 = np.sum(counts > 1)
    G2ms = []
    for m in ms:
        rolled = np.roll(counts,m)
        G2m = np.sum(counts*rolled >= 1)
        G2ms.append(G2m)
    G2ms = np.array(G2ms)
    G2m = np.mean(G2ms)
    #G20 = np.max([G20,B])
    g20 = (G20-B)/(G2m-B)
    sigma = compute_sigma(G20,G2m,len(ms),B)
    return g20,sigma

